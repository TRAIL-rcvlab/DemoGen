task=${1}
gen_range=${2}
gen_mode=${3}
n_gen_per_source=${4}
render_video=${5}

data_root=../data

repo_root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${repo_root}:${repo_root}/diffusion_policies:${PYTHONPATH}"

python -W ignore gen_demo.py --config-name=${task}.yaml \
                                data_root=${data_root} \
                                generation.range_name=${gen_range} \
                                generation.mode=${gen_mode} \
                                generation.n_gen_per_source=${n_gen_per_source} \
                                generation.render_video=${render_video}
                                     