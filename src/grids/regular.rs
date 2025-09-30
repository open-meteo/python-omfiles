use crate::grids::gaussian::GaussianGrid;

pub struct RegularGrid {
    pub nx: usize,
    pub ny: usize,
    pub lat_min: f32,
    pub lon_min: f32,
    pub dx: f32,
    pub dy: f32,
    pub search_radius: usize,
}

impl RegularGrid {
    pub fn new(
        nx: usize,
        ny: usize,
        lat_min: f32,
        lon_min: f32,
        dx: f32,
        dy: f32,
        search_radius: usize,
    ) -> Self {
        Self {
            nx,
            ny,
            lat_min,
            lon_min,
            dx,
            dy,
            search_radius,
        }
    }

    pub fn is_global(&self) -> bool {
        (self.nx as f32 * self.dx) >= 360.0 && (self.ny as f32 * self.dy) >= 180.0
    }

    pub fn find_point(&self, lat: f32, lon: f32) -> Option<usize> {
        let (x, y) = self.find_point_xy(lat, lon)?;
        Some(y * self.nx + x)
    }

    pub fn find_point_xy(&self, lat: f32, lon: f32) -> Option<(usize, usize)> {
        let x = ((lon - self.lon_min) / self.dx).round() as isize;
        let y = ((lat - self.lat_min) / self.dy).round() as isize;

        // Border handling (ICON global, dateline, etc.)
        let xx = if self.nx as f32 * self.dx >= 359.0 {
            if x == -1 {
                0
            } else if x == self.nx as isize || x == self.nx as isize + 1 {
                self.nx as isize - 1
            } else {
                x
            }
        } else {
            x
        };
        let yy = if self.ny as f32 * self.dy >= 179.0 {
            if y == -1 {
                0
            } else if y == self.ny as isize {
                self.ny as isize - 1
            } else {
                y
            }
        } else {
            y
        };

        if xx < 0 || yy < 0 || xx >= self.nx as isize || yy >= self.ny as isize {
            None
        } else {
            Some((xx as usize, yy as usize))
        }
    }

    pub fn get_coordinates(&self, gridpoint: usize) -> (f32, f32) {
        let y = gridpoint / self.nx;
        let x = gridpoint % self.nx;
        let lat = self.lat_min + y as f32 * self.dy;
        let lon = self.lon_min + x as f32 * self.dx;
        (lat, lon)
    }

    pub fn find_point_interpolated(&self, lat: f32, lon: f32) -> Option<GridPoint2DFraction> {
        let x = (lon - self.lon_min) / self.dx;
        let y = (lat - self.lat_min) / self.dy;

        if y < 0.0 || x < 0.0 || y >= self.ny as f32 || x >= self.nx as f32 {
            return None;
        }

        let x_fraction = (lon - self.lon_min) % self.dx;
        let y_fraction = (lat - self.lat_min) % self.dy;
        Some(GridPoint2DFraction {
            gridpoint: (y as usize) * self.nx + (x as usize),
            x_fraction,
            y_fraction,
        })
    }
}

pub struct GridPoint2DFraction {
    pub gridpoint: usize,
    pub x_fraction: f32,
    pub y_fraction: f32,
}
