

# <a href="https://demo-generation.github.io/">𝑫𝒆𝒎𝒐𝑮𝒆𝒏: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning</a>

<a href="https://demo-generation.github.io/"><strong>Project Page</strong></a> | <a href="https://arxiv.org/abs/2502.16932"><strong>arXiv</strong></a> | <a href="https://x.com/ZhengrongX/status/1899134914416800123"><strong>Twitter</strong></a> 

**Robotics: Science and Systems (RSS) 2025**


# 🎯 Overview

<div align="center">
  <img src="pics/teaser.png" alt="teaser" width="100%">
</div>

𝑫𝒆𝒎𝒐𝑮𝒆𝒏 is a synthetic data generation approach designed for robotic manipulation. Given only one human demonstration collcted in the real world, 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 could produce hundreds of spatially-augmented synthetic demonstrations in few seconds. These demos are proved to be effective for training visuomotor policies (e.g., [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy)) with strong O.O.D. generalization capabilities.

<br>
<div align="center">
  <img src="pics/method.png" alt="method" width="100%">
</div>
For action generation, 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 adopts the idea of Task and Motion Planning (TAMP) and adapts the source actions according to novel object configurations. For visual observation generation, 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 leverages 3D point clouds as the modality and rearranges the subjects in the scene via 3D editing.

# 🐣 Update
* **2025/04/02**, Officially released 𝑫𝒆𝒎𝒐𝑮𝒆𝒏.


# 🚀 Quick Try in 5 Minutes
## 1. Minimal Installation
#### 1.0. Create conda Env
```bash
conda remove -n demogen --all
conda create -n demogen python=3.8
conda activate demogen
```

#### 1.1. Install pip Packages 
```bash
pip3 install imageio imageio-ffmpeg termcolor hydra-core==1.2.0 zarr==2.12.0 matplotlib setuptools==59.5.0 pynput h5py scikit-video tqdm
```

#### 1.2. Install diffusion_policies
We only need the dataset loader in the diffusion_policies package.
```bash
cd diffusion_policies
pip install -e .
cd ..
```

#### 1.3. Install 𝑫𝒆𝒎𝒐𝑮𝒆𝒏
```bash
cd demo_generation
pip install -e .
cd ..
```

## 1.5. (Optional) Docker-based Installation

A `docker-compose.yml` is provided as an alternative to the conda setup.

```bash
# Start an interactive dev container
docker compose up -d dev
docker compose exec dev bash

# Or run a specific profile directly
docker compose --profile demogen run --rm demogen
docker compose --profile train    run --rm train
docker compose --profile inference run --rm inference
```

#### Proxy Configuration (for server users behind a firewall)

If your server requires an HTTP proxy (e.g., via an SSH reverse tunnel), export the
proxy variables in the **host** shell before starting the containers.  The
`docker-compose.yml` is configured to pass them through automatically:

```bash
# Example: SSH reverse tunnel forwarding a local SOCKS/HTTP proxy to port 1080
ssh -R 1080:localhost:7897 user@your-server

# On the server, export the proxy variables before starting Docker
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080

# Then start the container — the proxy settings will be available inside
docker compose up -d dev
```

You can also add the `export` lines to your `~/.bashrc` (with an availability
check) so they are set automatically on every login:

```bash
# Requires netcat (install with: sudo apt-get install netcat-openbsd)
if nc -z 127.0.0.1 1080 2>/dev/null; then
    export http_proxy=http://127.0.0.1:1080
    export https_proxy=http://127.0.0.1:1080
else
    unset http_proxy
    unset https_proxy
fi
```

If `nc` is not available, you can use `curl` as an alternative check:

```bash
if curl -s --max-time 1 http://127.0.0.1:1080 >/dev/null 2>&1 || \
   curl -s --max-time 1 --proxy http://127.0.0.1:1080 http://example.com >/dev/null 2>&1; then
    export http_proxy=http://127.0.0.1:1080
    export https_proxy=http://127.0.0.1:1080
else
    unset http_proxy
    unset https_proxy
fi
```

## 2. Generate Synthetic Demos Using 𝑫𝒆𝒎𝒐𝑮𝒆𝒏
#### 2.1. The 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 implementation
The 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 procedure is implemented in `demo_generation/demo_generation/demogen.py`. To run the code, you need to specify a `.yaml` config file under the `demo_generation/demo_generation/config` folder, where we provide some examples for your reference. The outer entrance that combines the main code and configs is `demo_generation/gen_demo.py`.

#### 2.2. Inputs & Outputs
We prepare some `.zarr` datasets consisting of 1~3 source demos under the folder `data/datasets/source`. By running the `gen_demo.py` script with proper config file, you can generate datsets of synthetic demos, which will be placed under the `data/datasets/generated` folder. To get a sense of what has been generated, you can check the rendered videos under the `data/videos` folder, when the `generation:render_video` flag is set to `True` in the config file. 

**Note:** While the demo generation process is very fast, it takes ~10s to render the video for a single generated trajectory. So we recommend rendering videos only for debugging purpose.

#### 2.3. Demo Generation!
We provide some example generation commands in the `demo_generation/run_gen_demo.sh` script, including four tasks: **Flower-Vase**, **Mug-Rack**, **Spatula-Egg**, and **Sauce-Spreading**. You can try running it and compare the results of synthetic and source demos in the `data/datasets/generated` and `data/videos` folders, where the filename of the videos indicate how the objects are transformed.
```bash
cd demo_generation
bash run_gen_demo.sh
```


# 🛠️ Run On Your Own Tasks
As long as your task requires to collect a handful of demonstrations to overcome the spatial generalization problem, 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 could be your remedy for saving the repetitive human labor. As is proved by the experiments we have conducted in our paper, 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 is generally effective for various types of tasks, even those involving contact-rich motion skills. To help you apply 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 to your own task, we prepare a detailed guide under the `docs` folder. Check it out if you are interested!


# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 👍 Acknowledgement
Our code is generally built upon: [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [UMI](https://github.com/real-stanford/universal_manipulation_interface), [MimicGen](https://github.com/NVlabs/mimicgen). We thank all these authors for their nicely open sourced code and their great contributions to the community.

Contact [Zhengrong Xue](https://steven-xzr.github.io/) if you have any questions or suggestions.

# 📝 Citation

If you find our work useful, please consider citing:
```
@article{xue2025demogen,
  title={DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning},
  author={Xue, Zhengrong and Deng, Shuying and Chen, Zhenyang and Wang, Yixuan and Yuan, Zhecheng and Xu, Huazhe},
  journal={arXiv preprint arXiv:2502.16932},
  year={2025}
}
```
