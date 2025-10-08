import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


class LocalStableDiffusionPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        method=None,
        args = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Ours (initial sample adjustment)
        if method == 'adj_init_noise':
            if args.per_sample:
                latents_list = []
                for sample_idx in range(num_images_per_prompt):
                    latent = latents[sample_idx:sample_idx+1]
                    uncond_prompt_embeds = prompt_embeds[sample_idx]
                    cond_prompt_embeds = prompt_embeds[sample_idx+num_images_per_prompt]
                    prompt_embed = torch.stack([uncond_prompt_embeds, cond_prompt_embeds], dim=0)
                    with torch.enable_grad():
                        latent = self.adj_init_noise_per_sample(
                            latents=latent,
                            prompt_embeds=prompt_embed,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            target_steps=[0],
                            lr=args.lr,
                            optim_iters=args.optim_iters,
                            target_loss=args.target_loss,
                            print_optim=True,
                        )
                    latents_list.append(latent)
                assert len(latents_list) == num_images_per_prompt
                latents = torch.cat(latents_list, dim=0)
            elif args.batch_wise:
                latents = self.adj_init_noise_batch_wise(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    adj_iters=args.adj_iters,
                    rho=args.rho,
                    gamma=args.gamma,
                )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if (method == 'adj_init_noise') and args.batch_wise:
                        if i <= args.apply_cfg_step:
                            noise_pred = noise_pred_uncond
                        else:
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)


        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
    
    def adj_init_noise_batch_wise(
        self,
        num_inference_steps=50,
        guidance_scale=7.0,
        latents=None,
        prompt_embeds=None,
        adj_iters=2,
        rho=50,
        gamma=0.7,
    ):
        if prompt_embeds == None:
            print("prompt_embeds are NONE!")
            import sys
            sys.exit()
        if latents == None:
            print("latents are NONE!")
            import sys
            sys.exit()

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for _ in range(adj_iters):
            latent_model_input = (torch.cat([latents] * 2))
            noise_pred = self.unet(
                latent_model_input,
                timesteps[0],
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            eps_tilde = noise_pred_text - noise_pred_uncond
            eps_tilde_mag = torch.sqrt(torch.sum(eps_tilde**2, dim=(1, 2, 3), keepdim=True))
            delta_hat = rho * (eps_tilde / eps_tilde_mag)
            latents_w_delta_hat = latents + delta_hat

            latent_model_input = (torch.cat([latents_w_delta_hat] * 2))
            noise_pred_w_delta_hat = self.unet(
                latent_model_input,
                timesteps[0],
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            noise_pred_uncond_w_delta_hat, noise_pred_text_w_delta_hat = noise_pred_w_delta_hat.chunk(2)
            eps_tilde_w_delta_hat = noise_pred_text_w_delta_hat - noise_pred_uncond_w_delta_hat
            nabla_l_sharp = eps_tilde_w_delta_hat - eps_tilde
            latents = latents - gamma * nabla_l_sharp
        return latents
    
    def adj_init_noise_per_sample(
        self,
        num_inference_steps=50,
        guidance_scale=7.0,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        target_steps=[0],
        lr=0.1,
        optim_iters=10,
        target_loss=None,
        print_optim=False,
    ):
        if prompt_embeds == None:
            print("prompt_embeds are NONE!")
            import sys
            sys.exit()
        if latents == None:
            print("latents are NONE!")
            import sys
            sys.exit()

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if i in target_steps:
                    latents.requires_grad = True
                    latents_optim = torch.optim.AdamW([latents], lr=lr)

                    for j in range(optim_iters):
                        latent_model_input = (
                            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        input_prompt_embeds = prompt_embeds

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=input_prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        loss_main = torch.norm(noise_pred_text, p=2).mean()
                        loss = loss_main

                        if target_loss is not None:
                            if loss.item() <= target_loss:
                                if print_optim is True:
                                    print(f"[Latent Optim] step: {j}, loss: {loss.item()}")
                                break

                        latents_optim.zero_grad()
                        loss.backward()
                        latents_optim.step()

                    latents = latents.detach()
                    latents.requires_grad = False
                    torch.cuda.empty_cache()
                    return latents
                
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                progress_bar.update()

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents_img2img(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
    ):
        # if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        #     raise ValueError(
        #         f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        #     )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            # deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents
