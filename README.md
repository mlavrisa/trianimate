# trianimate
Library for triangulation of images, and subsequent animation of the triangles.

Why use this triangulation instead of the gazillions of other options? I prefer the way it looks. The library aims to reduce the size of triangles, and only pick the most important edges, while still doing a decent job at picking up small or narrow features. It comes at an increased computational cost, but to me the result is worth it.

Currently just getting started, the triangulation is working nicely, but that's all that's here for now. More coming soon!

Eventually this will be a fully fledged package, with an editor for creating procedural animations using these triangulations, with a simple timeline feature. For larger projects, this tool should only be used to create video clips which can be imported into dedicated video editing software.

If anyone finds this useful, please let me know! It was intended for personal use, but I'm happy to make it available to others.

# To do
In summary, there's a lot to do. More specifically the roadmap looks like:
- user interface using pyqt
- static rendering in openGL
- export to image
- point selection to then apply animations
- apply basic library of animation styles
- export to video
- resample colours per frame during animations (or not)
- create and save custom animation styles
- set depth of points, to then allow...
- basic 3D animations