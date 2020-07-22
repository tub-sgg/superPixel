//
// Created by alxlee on 08.06.20.
//

#ifndef NOISE_SUPER_PIXEL_LABEL_NOISE_H
#define NOISE_SUPER_PIXEL_LABEL_NOISE_H

namespace SuperPixel {
    enum options {
        RandomSelectRandomRelabel,
        RandomSelectConfusionRelabel,
        RandomSelectNeighbourInReferenceRelabel,
        RandomSelectNeighbourInFeatureSpaceRelabel,

        UncertaintyAreaSelectRandomRelabel,
        UncertaintyAreaSelectConfusionRelabel,
        UncertaintyAreaSelectNeighbourInReferenceRelabel,
        UncertaintyAreaSelectNeighbourInFeatureSpaceRelabel,

        ObjectBorderSelectRandomRelabel,
        ObjectBorderSelectConfusionRelabel,
        ObjectBorderSelectNeighbourInReferenceRelabel,
        ObjectBorderSelectNeighbourInFeatureSpaceRelabel

    };

    enum PixelSelectionType {
        randomSelection, uncertaintyArea, objectBorder
    };
    enum NewLabelType {
        randomRelabel, CMrelabel, neighbourInReference, neighbourInFeatureSpace
    };

}
#endif //NOISE_SUPER_PIXEL_LABEL_NOISE_H
