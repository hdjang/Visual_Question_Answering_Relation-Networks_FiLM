CUDA_VISIBLE_DEVICES=4 \
python main.py  --exp_id       -                \
                --model        FiLM             \
                --phase        test             \
                --num_workers  40               \
                --batch_size   64               \
                \
                --film_cnn_chs 128,128,128,128  \
                \
                --checkpoint   './exp/film-exp-1/RN_epoch_40.pth' \
                \
                --extract_manifold_source