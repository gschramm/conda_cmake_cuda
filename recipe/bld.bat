cmake %RECIPE_DIR%\..\ -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=%PREFIX%
cmake --build . --target install
