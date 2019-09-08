CUDA_VISIBLE_DEVICES=1 \
python main.py  --exp_id       film-exp-1-re  \
                --model        FiLM           \
                --phase        train          \
                --num_workers  40             \
                --batch_size   64             \
                --epochs       60             \
                \
                --optimizer    adam           \
                --lr           0.0003         \
                --weight_decay 0.00001        \
                \
                #--resume_ckpt  './exp/film-exp-1-re/FiLM_epoch_50.pth' \