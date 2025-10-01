pub trait Grid2D {
    fn nx(&self) -> usize;
    fn ny(&self) -> usize;
    /// Get lat, lon coordinates for a given grid point
    fn get_coordinates_2d(&self, grid_x: usize, grid_y: usize) -> (f32, f32);
    /// Find nearest gridpoint index for (lat, lon)
    fn find_point_xy(&self, lat: f32, lon: f32) -> Option<(usize, usize)>;
}
