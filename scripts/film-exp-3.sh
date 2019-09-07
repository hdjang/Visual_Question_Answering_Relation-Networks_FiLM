CUDA_VISIBLE_DEVICES=2 \
python main.py  --exp_id       film-exp-3     \
                --model        FiLM           \
                --phase        train          \
                --num_workers  40             \
                --batch_size   64             \
                --epochs       50             \
                \
                --optimizer    adam           \
                --lr           0.0002         \
                --weight_decay 0.00001        \
                \