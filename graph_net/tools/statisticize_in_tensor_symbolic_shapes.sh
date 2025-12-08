#/bin/bash
bash get_in_tensor_symbolic_shapes.sh | grep get-in-tensor-symbolic-shapes | awk '{print $2}' | sort | uniq -c | sort -nk1
