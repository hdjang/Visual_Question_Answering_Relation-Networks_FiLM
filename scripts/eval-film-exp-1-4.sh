CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id       film-exp-1-4   \
                --model        FiLM           \
                --phase        test           \
                --num_workers  40             \
                --batch_size   64             \
                --epochs       50             \
                \
                --optimizer    adam           \
                --lr           0.0003         \
                --weight_decay 0.00001        \
                \
                --film_cnn_chs 32,64,128,128  \
                \
                --checkpoint   './exp/film-exp-1-4/RN_epoch_40.pth' \