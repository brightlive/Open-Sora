import os
import time
from pprint import pformat
from cog import BasePredictor, Input, Path
import sys
import subprocess
import tempfile

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
from google.cloud import storage
from PIL import Image
import random

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

def download_public_file(bucket_name, source_blob_name, destination_file_name):

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )

def repeat_frames(video_path, save_path, r, target_width, target_height):
    if r < 1:
        raise ValueError("Repeat factor 'r' must be 1 or greater.")

    # Check if the output path is the same as the input path
    if video_path == save_path:
        tmp_save_path = save_path + ".tmp.mp4"
    else:
        tmp_save_path = save_path

    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-i', video_path,                         # Input video
        '-vf', f"minterpolate='fps=24',scale={target_width}:{target_height}", # Frame repetition filter
        '-c:a', 'copy',  # Copy audio as is
        tmp_save_path,  # Output video path
    ]

    # Execute the command
    subprocess.run(command, check=True)

    # If a temporary file was used, overwrite the original file
    if video_path == save_path:
        subprocess.run(['mv', tmp_save_path, save_path], check=True)

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

class Predictor(BasePredictor):
    text_encoder = None
    vae = None
    def setup(self) -> None:

        # == init logger ==
        self.logger = create_logger()
        #self.logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
        verbose = 1
        self.progress_wrap = tqdm if verbose == 1 else (lambda x: x)

        # == device and dtype ==
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg_dtype = "bf16"
        assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
        self.dtype = to_torch_dtype("bf16")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # == init distributed env ==
        if is_distributed():
            colossalai.launch_from_torch({})
            self.coordinator = DistCoordinator()
            self.enable_sequence_parallelism = coordinator.world_size > 1
            if enable_sequence_parallelism:
                set_sequence_parallel_group(dist.group.WORLD)
        else:
            self.coordinator = None
            self.enable_sequence_parallelism = False

        # ======================================================
        # build model & load weights
        # ======================================================
        self.logger.info("Building models...")
        # == build text-encoder and vae ==
        if self.text_encoder == None:
            print("Didn't find text encoder, loading it")
            self.text_encoder = build_module(dict(
                type="t5",
                from_pretrained="models/t5",
                model_max_length=300,
            ), MODELS, device=self.device)
        else:
            print("Found pre-existing text encoder!")
        if self.vae == None:
            print("Didn't find vae, loading it")
            self.vae = build_module(dict(
                type="OpenSoraVAE_V1_2",
                from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
                micro_frame_size=17,
                micro_batch_size=4,
            ), MODELS).to(self.device, self.dtype).eval()
        else:
            print("Found pre-existing vae!")
    
    def predict(
        self,
        config: str = "/src/configs/opensora-v1-2/inference",
        prompt: str = None,
        width: int = 512,
        height: int = 512,
        steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=100,
            default=25,
        ),
        flow: float = Input(
            description="Flow. Strength of the motion",
            ge=0.0,
            le=20,
            default=0.25,
        ),
        fps: int = Input(default=8, ge=1, le=60),
        fps_output: int = Input(default=24, ge=1, le=60),
        video_length: int = Input(
            description="Length of the video in frames (playback is at 8 fps e.g. 16 frames @ 8 fps is 2 seconds)",
            default=24,
            ge=1,
            le=1024,
        ),
        referenceImg: str = Input(default=None),
        bucket_name: str = Input(default="bright-live-ai.appspot.com"),
        seed: int = Input(
            description="Seed for different images and reproducibility. Use -1 to randomise seed",
            default=-1,
        ),
        guidance_scale: float = Input(
            description="Guidance Scale. How closely do we want to adhere to the prompt and its contents",
            ge=0.0,
            le=20,
            default=7.5,
        ),
        aes: float = Input(default=6.5, ge=0, le=20),
        loop: int = Input(default=1, ge=0, le=10),
        condition_frame_length: int = Input(default=5, ge=0, le=20),
        # Unused
        condition_frame_edit: float = Input(default=0, ge=0, le=20),
        controlnetStrength: float = Input(default=0.1, ge=0.0, le=1.0),
        ipAdapterStrength: float = Input(default=0.5, ge=0.0, le=1.0),
        face: bool = Input(default=False),
        negative_prompt: str = Input(
            default="",
        ),
        ai_upscale: int = Input(
            description="AI Upscaler to use, if any. 0 for traditional upscaling, 1 for tile-upscale and 2 for refine",
            default=0,
            ge=0,
            le=2,
        ),
        path: str = Input(
            description="Choose the base model for animation generation. If 'CUSTOM' is selected, provide a custom model URL in the next parameter",
            default="toonyou_beta3.safetensors",
        ),

    ) -> Path:
        print("Here we are in predict")
        torch.set_grad_enabled(False)
        # ======================================================
        # configs & runtime variables
        # ======================================================
        # == parse configs ==
        if len(sys.argv) == 1:  # No arguments passed
            sys.argv.append("/src/configs/opensora-v1-2/inference/sample.py")
        cfg = parse_configs(training=False)

        repeat_factor = fps_output / fps

        generate_height = height
        generate_width = width

        if referenceImg is not None and referenceImg != "":
            img2video = True
            if bucket_name is not None and bucket_name != "":
                #os.system("mkdir input")
                # os.system("cp brian512.png input/00000000.png") #temp
                download_public_file(bucket_name, referenceImg, "input.png")
                referenceImg = "input.png"
            with Image.open(referenceImg) as img:
                generate_width, generate_height = img.size
                if generate_height > 1024:
                    resized_height = 1024
                    generate_width = int((resized_height / generate_height) * generate_width)
                    generate_height = resized_height
                    resized_img = img.resize((generate_width, generate_height), Image.BICUBIC)
                    resized_img.save(referenceImg)
        
        if generate_width > 1024 or generate_height > 1024 and False:
            generate_height = generate_height * 0.5
            generate_width = generate_width * 0.5
        cfg.image_size = (int(generate_height), int(generate_width)) # These coordinates must be presented backwards
        
        cfg.steps = steps
        # Temp
        # cfg.steps = 50

        cfg.loop = loop
        
        cfg.cfg_scale = guidance_scale

        # == device and dtype ==
        # now in setup

        # == init distributed env ==
        # now in setup
        
        #set_random_seed(seed=cfg.get("seed", 1024))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_random_seed(seed=seed)
        print(f"Using seed: {seed}")

        # == init logger ==
        # Now in setup

        # ======================================================
        # build model & load weights
        # ======================================================
        # Now in setup

        # == prepare video size ==
        
        image_size = cfg.get("image_size", None)
        #image_size = (width, height)
        if image_size is None:
            resolution = cfg.get("resolution", None)
            aspect_ratio = cfg.get("aspect_ratio", None)
            assert (
                resolution is not None and aspect_ratio is not None
            ), "resolution and aspect_ratio must be provided if image_size is not provided"
            image_size = get_image_size(resolution, aspect_ratio)
        #num_frames = get_num_frames(cfg.num_frames)
        num_frames = video_length

        # == build diffusion model ==
        input_size = (num_frames, *image_size)
        latent_size = self.vae.get_latent_size(input_size)
        model = (
            build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=self.vae.out_channels,
                caption_channels=self.text_encoder.output_dim,
                model_max_length=self.text_encoder.model_max_length,
                enable_sequence_parallelism=self.enable_sequence_parallelism,
            )
            .to(self.device, self.dtype)
            .eval()
        )
        self.text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

        # == build scheduler ==
        scheduler = build_module(cfg.scheduler, SCHEDULERS)

        # ======================================================
        # inference
        # ======================================================
        # == load prompts ==
        prompts = cfg.get("prompt", None)
        if prompt != None:
            prompts = [prompt]
        start_idx = cfg.get("start_index", 0)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

        # == prepare reference ==
        #reference_path = cfg.get("reference_path", [""] * len(prompts))
        if referenceImg != None:
            reference_path = [referenceImg]
            mask_strategy = ["0"]
        else:
            reference_path = [""]
            mask_strategy = [""]
        #mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
        
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==

        # My old way
        #save_fps = fps

        # Trying this
        save_fps = int(video_length / 24)


        #save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
        multi_resolution = cfg.get("multi_resolution", None)
        batch_size = cfg.get("batch_size", 1)
        num_sample = cfg.get("num_sample", 1)
        loop = cfg.get("loop", 1)
        condition_frame_length = condition_frame_length
        condition_frame_edit = condition_frame_edit
        align = cfg.get("align", None)

        save_dir = cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)
        sample_name = cfg.get("sample_name", None)
        prompt_as_path = cfg.get("prompt_as_path", False)

        # == Iter over all samples ==
        for i in self.progress_wrap(range(0, len(prompts), batch_size)):
            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = collect_references_batch(refs, self.vae, image_size)

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), image_size, num_frames, fps, self.device, self.dtype
            )

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = [
                    get_save_path_name(
                        save_dir,
                        sample_name=sample_name,
                        sample_idx=start_idx + idx,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    )
                    for idx in range(len(batch_prompts))
                ]

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if prompt_as_path and all_exists(save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # 1. refine prompt by openai
                if cfg.get("llm_refine", False):
                    # only call openai API when
                    # 1. seq parallel is not enabled
                    # 2. seq parallel is enabled and the process is rank 0
                    if not self.enable_sequence_parallelism or (self.enable_sequence_parallelism and is_main_process()):
                        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                            batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                    # sync the prompt if using seq parallel
                    if self.enable_sequence_parallelism:
                        self.coordinator.block_all()
                        prompt_segment_length = [
                            len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                        ]

                        # flatten the prompt segment list
                        batched_prompt_segment_list = [
                            prompt_segment
                            for prompt_segment_list in batched_prompt_segment_list
                            for prompt_segment in prompt_segment_list
                        ]

                        # create a list of size equal to world size
                        broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                        dist.broadcast_object_list(broadcast_obj_list, 0)

                        # recover the prompt list
                        batched_prompt_segment_list = []
                        segment_start_idx = 0
                        all_prompts = broadcast_obj_list[0]
                        for num_segment in prompt_segment_length:
                            batched_prompt_segment_list.append(
                                all_prompts[segment_start_idx : segment_start_idx + num_segment]
                            )
                            segment_start_idx += num_segment

                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes=aes,
                        flow=flow,
                        camera_motion=cfg.get("camera_motion", None),
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = append_generated(
                            self.vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    # == sampling ==
                    torch.manual_seed(1024)
                    z = torch.randn(len(batch_prompts), self.vae.out_channels, *latent_size, device=self.device, dtype=self.dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = scheduler.sample(
                        model,
                        self.text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=self.device,
                        additional_args=model_args,
                        progress=False,
                        mask=masks,
                    )
                    samples = self.vae.decode(samples.to(self.dtype), num_frames=num_frames)
                    video_clips.append(samples)

                # == save samples ==
                if is_main_process():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        if False:
                            self.logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[i][idx] for i in range(loop)]
                        for i in range(1, loop):
                            video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                        video = torch.cat(video, dim=1)
                        save_path = save_sample(
                            video,
                            fps=fps,
                            save_path=save_path,
                            verbose=False,
                        )
                        if save_path.endswith(".mp4") and cfg.get("watermark", False):
                            time.sleep(1)  # prevent loading previous generated video
                            add_watermark(save_path)
                        
                        if repeat_factor > 1 or height > generate_height:
                            time.sleep(1)  # prevent loading previous generated video
                            repeat_frames(save_path, save_path, repeat_factor, width, height)  

            start_idx += len(batch_prompts)
        self.logger.info("Inference finished.")
        self.logger.info("Saved %s samples to %s", start_idx, save_dir)
        return Path('samples/samples/sample_0000.mp4')