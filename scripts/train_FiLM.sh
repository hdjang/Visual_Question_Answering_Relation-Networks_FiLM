CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id       FiLM           \
                --model        FiLM           \
                --phase        train          \
                --num_workers  40             \
                --batch_size   64             \
                --epochs       55             \
                \
                --optimizer    adam           \
                --lr           0.0003         \
                --weight_decay 0.00001        \
