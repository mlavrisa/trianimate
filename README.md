# trianimate
Library for triangulation of images, and subsequent animation of the triangles.

Why make _this_ triangulation instead of using the gazillions of other options? I prefer the way it looks, and the flexibility it offers. The method aims to reduce the number of overly small triangles, and only pick the most important edges, while still doing a decent job at capturing small or narrow features. It comes at an increased computational cost, but to me the result is worth it. The point picking algorithm is currently calculated with the help of numba to speed up execution. I'm not sure if C/C++ would add much speed benefit over this, but I might try it some day.

Currently this tool only has the ability to import an image, apply the triangulation procedure, and save the result. Much more is yet to come!

Eventually this will be a fully fledged package, with an editor for creating procedural animations using these triangulations, with the ability to preview the result before exporting to common video formats at any desired resolution. I'm planning to build a pick-and-place modular animation generator utility, which can modulate the positions of the vertices in myriad and highly customizable ways. For larger projects, this tool should only be used to create video clips which can be imported into dedicated video editing software.

If anyone finds this useful, please let me know! It was intended for personal use, but I'm happy to make it available to others. Issues and pull requests are welcome.

# To do
**In summary**: lots.

*More specifically* the roadmap in roughly the right order looks like:
- selection of vertices in the triangulation
    - lasso selection first
        - hold control to add to selection, alt to subtract
    - "magic wand" selection of vertices, based on colour similarity
- apply animations by selecting "Path", "PathTransform" and "TimeWarp" components and linking them together to create a final animation
    - components may use time and space as their input parameters, as well as additional user-controlled parameters. Typically based on mathematical functions and curves (Bezier curves, Fourier series, Lissajous curves, sinusoids, distance from point, distance from line, direction towards point, add, subtract, multiply, etc.) these functions are chained together in any allowed way to create a wide range of effects
- preview animations, and export to video
- recalculate triangulation during animation (or not. Although might look very strange and glitchy if not, especially at the edges, we'll see what happens)
- resample colours during animation (or not - and not is only an option if not recalculating triangles, otherwise the arrays could be different lengths, leading to errors)
- subclass the Path, PathTransform and TimeWarp classes to produce custom components, which can then be imported into the animations tool
- set the depth of points, to then allow...
- basic 3D animations
    - basically panning and rotating the camera
- separate groups of triangles into "layers", calculate triangulation for missing interior to allow for cleaner 3D effects, the above works best for gently varying depths
- remove triangles from the scene to produce transparent points in the exported video
