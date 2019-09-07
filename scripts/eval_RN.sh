CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id      -              \
                --model       RN             \
                --phase       test           \
                --num_workers 20             \
                --batch_size  128            \
                \
                --cnn_chs     32,64,128,256  \
                --rn_g_chs    512,512        \
                --rn_f_chs    512,512        \
                \
                --checkpoint  './model/RN.pth' \
                