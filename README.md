# SlicerDWD

Shape Analysis with DWD (Distance Weighted Discrimination), using the [`dwd` Python Package][dwd]. For details see
([Marron et al 2007][marron-et-al], [Wang and Zou 2018][wang-zou]).

![DWD Direction Visualization][vectors]

[dwd]: https://github.com/slicersalt/dwd
[vectors]: https://raw.githubusercontent.com/slicersalt/SlicerDWD/main/img/Vectors.png

[marron-et-al]: https://amstat.tandfonline.com/doi/abs/10.1198/016214507000001120
[wang-zou]: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12244

# Installation

The extension should be built as a 3D Slicer extension. The extension depends on [ShapeVariationAnalyzer][sva] and [ShapePopulationViewer][spv] for dataset creation and certain visualizations.

[sva]: https://github.com/DCBIA-OrthoLab/ShapeVariationAnalyzer
[spv]: https://github.com/NIRALUser/ShapePopulationViewer

# Documentation

See [usage.md][usage] for usage documentation.

[usage]: ./docs/usage.md
