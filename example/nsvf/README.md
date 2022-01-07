# NSVF Example

NSVF is one of influential extended works based on NeRF. It incorperate explicit vol representation (of learnt features) with implicit MLP representation to provide the rendering speedups as well as editability of the scene.

However, the NSVF's original implementation is a bit complicated for beginners. We decided to re-implement NSVF here based on Accel-RF's workflow. 

**Note**: this NSVF may not be a 100% faithful re-implementation of the original NSVF, but it should be fully functional and efficient enough.

Work in progress...

# Discusssion

List accel-RF's 5 module to see what modification we need to make… 

## Dataset

- Things in common: 

- - Intrinsics <=> hwf
  - Extrinsics <=> pose

- *Optional*

- - read image on the fly,  instead of pre-loading in memory
  - Alpha can be used as the mask.. (though this feature is not used in NSVF example
  - Trainable extrinsics is also interesting. (though not used.

## RaySampler

- Needs to be added

- - Add option Multiple views per batch

  - Sample rays based on intersection results.

  - - Voxel intersection test is also a feature to be added.

## PointSampler

- NSVF code has a  `inverse_cdf_sampling` function to do this.
- It seems that nsvf does not require hierarchical sampling.

## NN Models

- Positional Encoding

  - Add a angular option. 

- MLP models

- - NSVF has a special background field..

## Rendering

- Early termination mechanism

## 3D Representation (**New TODO**)

- Manage explicit 3D     representations.
  * [ ] Voxel grids
  * [ ] Octrees
  * [ ] …

- Support related 3D operations
  * [ ] Intersection
  * [ ] Pruning
  * [ ] Splitting
  * [ ] …

- Support differentialable features attached to the 3D representations.