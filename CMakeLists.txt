cmake_minimum_required(VERSION 3.13.4)

project(SlicerDWD)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerDWD")
set(EXTENSION_CATEGORY "Shape Analysis")
set(EXTENSION_CONTRIBUTORS "David Allemang (Kitware Inc.)")
set(EXTENSION_DESCRIPTION "Perform and visualize Distance-Weighted Discrimination on correspondent populations.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/slicersalt/SlicerDWD/main/DWD/Resources/Icons/SlicerDWD.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/slicersalt/SlicerDWD/main/img/Vectors.png")
set(EXTENSION_DEPENDS "ShapePopulationViewer ShapeVariationAnalyzer") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerDWD)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
