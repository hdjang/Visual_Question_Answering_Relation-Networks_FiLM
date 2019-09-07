CUDA_VISIBLE_DEVICES=3 \
python main.py  --exp_id       film-exp-1-3   \
                --model        FiLM           \
                --phase        train          \
                --num_workers  40             \
                --batch_size   64             \
                --epochs       50             \
                \
                --optimizer    adam           \
                --lr           0.0003         \
                --weight_decay 0.00001        \
                \
                --film_cnn_chs 16,32,64,128   \
                \
                #--resume_ckpt  './exp/film-exp-1/RN_epoch_20.pth' \