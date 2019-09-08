CUDA_VISIBLE_DEVICES=0 \
python main.py  --exp_id       -                \
                --model        FiLM             \
                --phase        test             \
                --num_workers  40               \
                --batch_size   128              \
                \
                --checkpoint   './model/FiLM.pth' \
                \
                --extract_manifold_source