CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id      -                \
                --model       FiLM             \
                --phase       test             \
                --num_workers 40               \
                --batch_size  64               \
                \
                --checkpoint  './model/FiLM.pth' \
                