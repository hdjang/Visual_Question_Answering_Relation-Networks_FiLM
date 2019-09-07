CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id       film-exp-1-2   \
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
                --cnn_feat_h   6              \
                --cnn_feat_w   6              \
                \
                #--resume_ckpt  './exp/film-exp-1/RN_epoch_20.pth' \