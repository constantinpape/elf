import elf.htm as htm


# the mobie-htm example project created with
# https://github.com/mobie/mobie-utils-python/blob/master/examples/create_mobie_htm_project.ipynb
(image_data, label_data,
 image_settings, label_settings,
 well_measurements, well_shape) = htm.parse_mobie("/home/pape/Work/data/mobie/mobie_htm_project/data",
                                                  dataset="example-dataset", view_name="default")
htm.view_plate(
    image_data, label_data,
    image_settings, label_settings,
    well_measurements, well_shape
)
