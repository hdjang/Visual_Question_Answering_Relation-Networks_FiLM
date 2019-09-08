CUDA_VISIBLE_DEVICES=5 \
python main.py  --exp_id       film-exp-1-re  \
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
                --checkpoint   './exp/film-exp-1-re/FiLM_epoch_50-Copy1.pth' \
                #--resume_ckpt  './exp/film-exp-1/RN_epoch_20.pth' \