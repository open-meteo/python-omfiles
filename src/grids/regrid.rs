use numpy::ndarray::Array2;

use crate::grids::{gaussian::GaussianGrid, regular::RegularGrid};

pub fn regrid_nearest(
    gaussian_grid: &GaussianGrid,
    gaussian_data: &[f32],
    regular_grid: &RegularGrid,
) -> Array2<f32> {
    let mut result = Array2::<f32>::zeros((regular_grid.ny, regular_grid.nx));

    for y in 0..regular_grid.ny {
        for x in 0..regular_grid.nx {
            let lat = regular_grid.lat_min + y as f32 * regular_grid.dy;
            let lon = regular_grid.lon_min + x as f32 * regular_grid.dx;

            // Find the nearest grid point in the Gaussian grid
            if let Some(idx) = gaussian_grid.find_point(lat, lon) {
                result[[y, x]] = gaussian_data[idx];
            } else {
                result[[y, x]] = f32::NAN; // or any missing value
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use crate::grids::gaussian::GaussianGrid;

    use numpy::ndarray::Axis;
    use omfiles_rs::{
        reader::OmFileReader,
        traits::{OmArrayVariable, OmFileReadable},
        writer::OmFileWriter,
    };

    use super::*;

    #[test]
    fn test_regrid_nearest() {
        // Create a tiny Gaussian grid (for test purposes)
        let gaussian_grid = GaussianGrid::new();

        let reader = OmFileReader::from_file(&"2025-09-30T0600.om").unwrap();
        let reader = reader.get_child_by_name("temperature_2m").unwrap();
        let reader = reader.expect_array().unwrap();
        println!("dimensions: {:?}", reader.get_dimensions());
        let read_range = &[0..reader.get_dimensions()[0], 0..reader.get_dimensions()[1]];
        // Dummy data: value = gridpoint index as f32
        let gaussian_data = reader.read::<f32>(read_range).unwrap();
        let gaussian_data = gaussian_data.as_slice().unwrap();

        let min = gaussian_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = gaussian_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        println!("Min: {}, Max: {}", min, max);

        let lat_steps = 1798;

        // Regular grid: global grid with 0.1 degree resolution
        let regular_grid = RegularGrid::new(
            3600, lat_steps, // nx, ny (360/0.1, 180/0.1)
            -89.9, -180.0, // lat_min, lon_min (global coverage)
            0.1, 0.1, // dx, dy
            1,
        );

        // Run bilinear regridding
        let result = regrid_nearest(&gaussian_grid, &gaussian_data, &regular_grid);
        assert_eq!(result.shape(), [lat_steps, 3600]);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut min_idx = (0, 0);
        let mut max_idx = (0, 0);

        for (i, row) in result.outer_iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val < min {
                    min = val;
                    min_idx = (i, j);
                }
                if val > max {
                    max = val;
                    max_idx = (i, j);
                }
            }
        }
        println!(
            "Min: {}, Max: {}, Min idx: {:?}, Max idx: {:?}",
            min, max, min_idx, max_idx
        );
        let result = result.insert_axis(Axis(2));
        assert_eq!(result.shape(), [lat_steps, 3600, 1]);

        // assert_eq!(result[[0, 20]], 2.0);

        // save output file
        let file_handle = File::create("output.om").unwrap();
        let mut writer = OmFileWriter::new(file_handle, 8 * 1024);

        let mut array_writer = writer
            .prepare_array(
                vec![lat_steps as u64, 3600, 1],
                vec![100, 100, 1],
                omfiles_rs::OmCompressionType::PforDelta2d,
                20.0,
                0.0,
            )
            .unwrap();

        array_writer
            .write_data(result.view().into_dyn(), None, None)
            .unwrap();

        let finalized = array_writer.finalize();

        let written_var = writer.write_array(finalized, "regridded", &[]).unwrap();

        writer.write_trailer(written_var);
    }
}
