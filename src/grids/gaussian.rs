use std::f32;

const O1280_LATITUDE_LINES: usize = 1280;
const O1280_COUNT: usize = 4 * 1280 * (1280 + 9); // 6599680

#[derive(Clone, Copy, Debug)]
pub enum GridType {
    O1280,
}

impl GridType {
    pub fn count(&self) -> usize {
        match self {
            GridType::O1280 => O1280_COUNT,
        }
    }

    pub fn latitude_lines(&self) -> usize {
        match self {
            GridType::O1280 => O1280_LATITUDE_LINES,
        }
    }

    pub fn nx_of(&self, y: usize) -> usize {
        let latitude_lines = self.latitude_lines();
        if y < latitude_lines {
            20 + y * 4
        } else {
            (2 * latitude_lines - y - 1) * 4 + 20
        }
    }

    pub fn integral(&self, y: usize) -> usize {
        let latitude_lines = self.latitude_lines();
        if y < latitude_lines {
            2 * y * y + 18 * y
        } else {
            let n = 2 * latitude_lines - y;
            self.count() - (2 * n * n + 18 * n)
        }
    }

    pub fn get_pos(&self, gridpoint: usize) -> (usize, usize, usize) {
        let latitude_lines = self.latitude_lines();
        let count = self.count();
        let half_count = count / 2;

        let y = if gridpoint < half_count {
            let gp = gridpoint as f32;
            let y_f = ((2.0 * gp + 81.0).sqrt() - 9.0) / 2.0;
            y_f.floor() as usize
        } else {
            let gp = (count - gridpoint - 1) as f32;
            let y_f = ((2.0 * gp + 81.0).sqrt() - 9.0) / 2.0;
            2 * latitude_lines - 1 - y_f.floor() as usize
        };

        let x = gridpoint - self.integral(y);
        let nx = self.nx_of(y);
        (y, x, nx)
    }
}

pub struct GaussianGrid {
    pub grid_type: GridType,
}

impl GaussianGrid {
    pub fn new() -> Self {
        Self {
            grid_type: GridType::O1280,
        }
    }

    pub fn nx(&self) -> usize {
        self.grid_type.count()
    }

    pub fn ny(&self) -> usize {
        1
    }

    pub fn nx_of(&self, y: usize) -> usize {
        self.grid_type.nx_of(y)
    }

    pub fn integral(&self, y: usize) -> usize {
        self.grid_type.integral(y)
    }

    pub fn get_coordinates(&self, gridpoint: usize) -> (f32, f32) {
        let latitude_lines = self.grid_type.latitude_lines();
        let (y, x, nx) = self.grid_type.get_pos(gridpoint);
        let dx = 360.0 / nx as f32;
        let dy = 180.0 / (2.0 * latitude_lines as f32 + 0.5);
        let lon = x as f32 * dx;
        let lat = (latitude_lines as f32 - y as f32 - 1.0) * dy + dy / 2.0;
        let lon = if lon >= 180.0 { lon - 360.0 } else { lon };
        (lat, lon)
    }

    pub fn find_point(&self, lat: f32, lon: f32) -> Option<usize> {
        let latitude_lines = self.grid_type.latitude_lines();
        let dy = 180.0 / (2.0 * latitude_lines as f32 + 0.5);
        let y_f =
            (latitude_lines as f32 - 1.0 - ((lat - dy / 2.0) / dy)) + 2.0 * latitude_lines as f32;
        let y = (y_f.round() as usize) % (2 * latitude_lines);

        let nx = self.nx_of(y);
        let dx = 360.0 / nx as f32;
        let x_f = (lon / dx) + nx as f32;
        let x = (x_f.round() as usize) % nx;

        Some(self.integral(y) + x)
    }
}
