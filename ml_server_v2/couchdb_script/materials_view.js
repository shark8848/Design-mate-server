function(doc) {
  if (doc.floors) {
    doc.floors.forEach(function (floor) {
      if (floor.houses) {
        floor.houses.forEach(function (house) {
          if (house.rooms) {
            house.rooms.forEach(function (room) {
              // Exterior Wall Material
              if (room.walls) {
                room.walls.forEach(function (wall) {
                  if( wall.insulation_material) {
                    emit([doc._id,wall.insulation_material.name, 'exterior_wall'], {
                      area: wall.material_area,
                      price: wall.insulation_material.price,
                      cost: wall.material_area * wall.insulation_material.price
                    });
                  }
                    

                  // Glass Material
                  if (wall.window && wall.window.glass_material) {
                    emit([doc._id,wall.window.glass_material.descriptions, 'glass'], {
                      area: wall.window.glass_area,
                      price: wall.window.glass_material.price,
                      cost: wall.window.glass_area * wall.window.glass_material.price
                    });

                    // Window Frame Material
                    
                    if (wall.window.wf_material) {
                      emit([doc._id,wall.window.wf_material.type, 'window_frame'], {
                        area: wall.window.wf_area,
                        price: wall.window.wf_material.price,
                        cost: wall.window.wf_area * wall.window.wf_material.price
                      });
                    }
                  }
                });
              }
            });
          }
        });
      }
    });
  }
}
