meson setup --reconfigure build_dir
meson compile -C ./build_dir

./build_dir/parameter
dot -Tpng graph.dot -o graph.png
open graph.png # macOS
