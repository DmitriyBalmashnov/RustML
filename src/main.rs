use std::{path::Path};
extern crate serde;
extern crate csv;
use serde::Deserialize;
use std::convert::TryInto;
mod mat;
mod linreg;
mod optimizer;
use mat::{Matrix, Vector};
use linreg::{LinearRegressor};
use optimizer::{Optimizer};

#[derive(Deserialize, Debug)]
struct HouseData {
    price: i32,
    area: i32,
    bedrooms: i32,
    bathrooms: i32,
    stories: i32
}

struct Houses<const N: usize> {
    data: [HouseData; N]
}

impl<const N: usize> Houses<N> {
    pub fn to_xy_data(&self) -> ([[f64; 5]; N], [[f64; 1]; N]) {
        let mut x = [[0.0; 5]; N];
        let mut y = [[0.0; 1]; N];

        for i in 0..N {
            y[i][0] = self.data[i].price as f64;
            x[i][0] = self.data[i].area as f64;
            x[i][1] = self.data[i].bedrooms as f64;
            x[i][2] = self.data[i].bathrooms as f64;
            x[i][3] = self.data[i].stories as f64;
            x[i][4] = 1.0;
        }
        return (x, y);
    }
}

fn house_vec_to_arr<const N: usize>(v: Vec<HouseData>) -> [HouseData; N] {
    v.try_into().unwrap_or_else(|_: Vec<HouseData>| panic!("Fucked up"))
}

fn read_houses(path: &str) -> Result<Vec<HouseData>, csv::Error> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    let records = rdr.deserialize::<HouseData>().filter_map(|s| s.ok()).collect::<Vec<HouseData>>();

    return Ok(records);
}

fn main() {
    let path = Path::new(".").join("data").join("Housing.csv").into_os_string().into_string().unwrap();
    let houses = match read_houses(&path) {
        Ok(houses) => houses,
        _ => panic!("file doesn't exist"),
    };
    println!("{:?}, {}", houses[0], houses.len());
    let housedata = Houses {data: house_vec_to_arr::<545>(houses)};
    let (x_data, y_data) = housedata.to_xy_data();
    let x = Matrix::from_data(x_data);
    let y = Vector::from_data(y_data);
    let lin_reg = LinearRegressor::train(&x, &y, Optimizer::PseudoInverse);

    let expected_cost = lin_reg.predict(&Vector::from_array([7420.0, 4.0, 2.0, 3.0, 1.0]));
    println!("Expected cost: {}", expected_cost)
}
