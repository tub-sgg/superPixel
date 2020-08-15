//
// Created by alxlee on 08.06.20.
//

#ifndef NOISE_SUPER_PIXEL_LABEL_NOISE_H
#define NOISE_SUPER_PIXEL_LABEL_NOISE_H

namespace SuperPixel {
    enum options {
        RandomSelectRandomRelabel,
        RandomSelectConfusionRelabel,
        RandomSelectGlobalNeighbourRelabel,
        RandomSelectNearestNeighbourRelabel,
        RandomSelectLocalNeighbourRelabel,

        UncertaintyAreaSelectRandomRelabel,
        UncertaintyAreaSelectConfusionRelabel,
        UncertaintyAreaSelectGlobalNeighbourRelabel,
        UncertaintyAreaSelectNearestNeighbourRelabel,
        UncertaintyAreaSelectLocalNeighbourRelabel,

        ObjectBorderSelectRandomRelabel,
        ObjectBorderSelectConfusionRelabel,
        ObjectBorderSelectGlobalNeighbourRelabel,
        ObjectBorderSelectNearestNeighbourRelabel,
        ObjectBorderSelectLocalNeighbourRelabel,

        RandomSelectionAllRelabel,
        UncertaintySelectAllRelabel,
        ObjectBorderSelectAllRelabel

    };

    enum PixelSelectionType {
        randomSelection, uncertaintyArea, objectBorder
    };
    enum NewLabelType {
        randomRelabel, CMrelabel, GlobalNeighbour, localNeighbour,NearestNeighbour
    };

}
#endif //NOISE_SUPER_PIXEL_LABEL_NOISE_H
