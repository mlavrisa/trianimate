# trianimate
Library for triangulation of images, and subsequent animation of the triangles.

Why make _this_ triangulation instead of using the gazillions of other options? I prefer the way it looks, and the flexibility it offers. The library aims to reduce the number of overly small triangles, and only pick the most important edges, while still doing a decent job at capturing small or narrow features. It comes at an increased computational cost, but to me the result is worth it. One day I may implement the point picking algorithm in C/C++ to improve the speed.

Currently this tool only has the ability to import an image, apply the triangulation procedure, and save the result. Much more is yet to come!

Eventually this will be a fully fledged package, with an editor for creating procedural animations using these triangulations, with the ability to preview the result before exporting to common video formats at any desired resolution. It may have a small timeline feature in the future, we'll see. For larger projects, this tool should only be used to create video clips which can be imported into dedicated video editing software.

If anyone finds this useful, please let me know! It was intended for personal use, but I'm happy to make it available to others. Issues and pull requests are welcome.

# To do
**In summary**: lots.

*More specifically* the roadmap in roughly the right order looks like:
- selection of vertices in the triangulation
    - lasso selection first
    - hold control to negate selection, shift to add 
    - "magic wand" selection of vertices, based on colour similarity
- apply a basic library of animation styles, with several options
    - primarily consist of various functions whose displacements from center are parametric in time (Fourier, Lissajous, Bezier...), whose net amplitudes are piecewise parametric in time (ramp up, hold, ramp down) and parametric in space and time (eg function of a distance from point, from line, etc.), and whose time parameter is parametric in time and space (Fourier, etc.)
- preview animations, and export to video
- recalculate triangulation during animation (or not. Although might look very strange and glitchy if not, especially at the edges, we'll see what happens)
- resample colours during animation (or not - and not is only an option if not recalculating triangles, otherwise the arrays will be different lengths)
- supply app with python functions which are parametric in time to calculate the point offsets, with custom checkboxes and sliders
- set depth of points, to then allow...
- basic 3D animations
    - basically panning and rotating the camera
- separate groups of triangles into "layers", calculate triangulation for missing interior to allow for cleaner 3D effects, the above works best for gently varying depths
