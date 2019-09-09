CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id      -                \
                --model       RN               \
                --phase       test             \
                --num_workers 40               \
                --batch_size  128              \
                \
                --cnn_chs     128,128,128,128  \
                --rn_g_chs    512,512          \
                --rn_f_chs    512,512          \
                \
                --rn_extension                 \
                --checkpoint  './model/RN_extension.pth' \
                