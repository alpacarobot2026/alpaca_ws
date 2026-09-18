import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from celluloid import Camera
import collections


# Copyright 2023 The human_scene_transformer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates Model on JRDB dataset."""

import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
import gin
from human_scene_transformer.jrdb import dataset_params
from human_scene_transformer.jrdb import input_fn
from human_scene_transformer.metrics import metrics
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params
import tensorflow as tf
import tqdm

# How it works
# 'has_data':  mask for all features. True at fields where the
# feature has data. Further, compute the global 'has_data' mask. True where
# xyz data are available.
#
# 'has_historic_data':  Further, compute the 'has_historic_data' mask.
# True for agents which have at least one valid xyz data point in the xyz feature.
#
# `is_padded`:  bool tensor of shape [batch (b), num_agents (a), num_timesteps (t), 1].
# True if the position is padded, ie, no valid observation.

# `should_predict`: which positions need to be predicted and save it to the `should_predict` bool tensor of shape [b, a, t, 1].
# A position should be predicted if it is hidden, not padded and the agent has historic data



_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    '/home/user/socialnav/jrdb_challenge_checkpoint/',
    'Path to model directory.',
)

_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    '/home/user/socialnav/jrdb_challenge_checkpoint/ckpts/ckpt-20',
    'Path to model checkpoint.',
)

# _MODEL_PATH = flags.DEFINE_string(
#     'model_path',
#     '/home/user/socialnav/jrdb_original_checkpoint/jrdb',
#     'Path to model directory.',
# )

# _CKPT_PATH = flags.DEFINE_string(
#     'checkpoint_path',
#     '/home/user/socialnav/jrdb_original_checkpoint/jrdb/ckpts/ckpt-30',
#     'Path to model checkpoint.',
# )

# #######################
def distance_error(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
  return tf.sqrt(
      tf.reduce_sum(tf.square(pred - target), axis=-1, keepdims=True))
# #######################


def evaluation(checkpoint_path):
  """Evaluates Model on Pedestrian dataset."""
  d_params = dataset_params.JRDBDatasetParams(num_agents=None)

  # only test on the specific scene ##################
  d_params.eval_scenes = d_params.eval_scenes[1:2]
  ######################################################

  print('Visualization Scene:', d_params.eval_scenes)
  dataset = input_fn.load_dataset(
      d_params,
      d_params.eval_scenes,
      augment=False,
      shuffle=False,
      allow_parallel=False,
      evaluation=False,
      repeat=False,
      keep_subsamples=False,
  )

  model_p = model_params.ModelParams()

  model = hst_model.HumanTrajectorySceneTransformer(model_p)

  _, _ = model(next(iter(dataset.batch(1))), training=False)

  checkpoint_mngr = tf.train.Checkpoint(model=model)
  checkpoint_mngr.restore(checkpoint_path).assert_existing_objects_matched()
  logging.info('Restored checkpoint: %s', checkpoint_path)


  batch_list = list(dataset.batch(1).as_numpy_iterator() )
  print(batch_list[0]['scene/id'][0], "iter size:", len(batch_list))
  position_shape = batch_list[0]['agents/position'].shape
  # input
  # agents/position: tf.Tensor: shape=(1, 11, 19, 2)
  # ...
  # input_batch: A dictionary that maps a str to a tensor. The tensor's first
        # dimensions is always the batch dimension. These tensors include all
        # timesteps (history, current and future) and all agents (observed and
        # padded).

  # output
  # output: A dict containing the model prediction. Note that the predicted
        # tensors has the same shape as the input_batch so the history and
        # current steps are included.
  # agents/position: tf.Tensor: shape=(1, 11, 19, 6, 2)
  # mixture_logits: tf.Tensor: shape=(1, 1, 1, 6)

  # should_predict: tf.Tensor: shape=(1, 11, 19, 1)   [batch, agent, time, value]
  ade_metric = metrics.ade.MinADE(model_p)
  ade_metric_1s = metrics.ade.MinADE(model_p, cutoff_seconds=1.0, at_cutoff=True)
  ade_metric_2s = metrics.ade.MinADE(model_p, cutoff_seconds=2.0, at_cutoff=True)
  ade_metric_3s = metrics.ade.MinADE(model_p, cutoff_seconds=3.0, at_cutoff=True)
  ade_metric_4s = metrics.ade.MinADE(model_p, cutoff_seconds=4.0, at_cutoff=True)

  ade_metric_plot = metrics.ade.MinADE(model_p, cutoff_seconds=0.0, at_cutoff=True)


  groundtrue_list = []
  pred_list = []
  robot_list = []
  robot_orientation_list = []

  mixture_logits_list = []

  test_instance = ade_metric_plot
  cutoff_idx = test_instance.cutoff_idx

  fig0, ax0 = plt.subplots()

  camera = Camera(fig0)
  print("Camera Snapping")
  step = 0

  red = (1,0,0,1)
  blue = (0,0,1,0.2)
  robot_color = (1,0,1,1)

  for input_batch in tqdm.tqdm(dataset.batch(1)):
    full_pred, output_batch = model(input_batch, training=False)

    # tqdm.tqdm.write(f"{output_batch['scene/id']}, {output_batch['scene/timestamp']}")
    scene_id = output_batch['scene/id']
    scene_t = output_batch['scene/timestamp']
    robot_position = output_batch['robot/position'][...,:full_pred['agents/position'].shape[-1]]       # [b, t, 2]
    robot_orientation = output_batch['robot/orientation']  # [b, t, 1]

    prediction = full_pred['agents/position']                    # [b, a, t, n, 2]
    scene_mixture_logits = full_pred['mixture_logits']

    should_predict = tf.cast(output_batch['should_predict'], tf.float32)  # [b, a, t, 1]
    target = output_batch['agents/position/target']                        # [b, a, t, 2]
    target = target[..., :full_pred['agents/position'].shape[-1]]

    # 'has data' means the data is provided by dataset (pedestrians exist)
    # ground true and prediction
    shape_b, shape_a, shape_t, shape_n, _ = prediction.shape
    tqdm.tqdm.write(f'b:{shape_b}, a:{shape_a}, t:{shape_t}, n:{shape_n}')

    current_t_cutoff = ade_metric_plot.cutoff_idx
    has_data = output_batch['has_data']   # [b, a, t, 1]  all data recorded from the dataset
    has_current_data = has_data[:, :, current_t_cutoff-1:current_t_cutoff, :]     # [b, a, 1, 1]
    has_current_data = tf.squeeze(has_current_data, [-1,-2])    # [b, a]

    groundtrue = tf.boolean_mask(target, has_current_data)
    groundtrue = tf.reshape(groundtrue, (shape_b, -1, shape_t, 2))    # [b, a-, t, 2]
    groundtrue_cur = groundtrue[:, :, current_t_cutoff-1:current_t_cutoff, :]           # [b, a-, 1, 2]

    # predict all current pedestrian's future trajectories
    pred = tf.boolean_mask(prediction, has_current_data)
    pred = tf.reshape(pred, (shape_b, -1, shape_t, shape_n, 2))    # [b, a-, t, n, 2]
    pred_future = pred[:, :, current_t_cutoff:, :, :]   # [b, a-, t-, n, 2]

    robot_cur = robot_position[:, current_t_cutoff-1:current_t_cutoff, :]   # [b, 1, 2]
    robot_orientation_cur = robot_orientation[:, current_t_cutoff-1:current_t_cutoff, :]   # [b, 1, 1]

    groundtrue_list.append(groundtrue_cur)
    pred_list.append(pred_future)
    robot_list.append(robot_cur)
    robot_orientation_list.append(robot_orientation_cur)
    # if test_instance.at_cutoff and test_instance.cutoff_seconds is not None:
    #   target = target[:, :, test_instance.cutoff_idx-1:test_instance.cutoff_idx, :]           # [b, a, t, 2]
    #   prediction = prediction[:, :, test_instance.cutoff_idx-1:test_instance.cutoff_idx, :]   # [b, a, t, n, 2]
    #   should_target = should_predict[:, :, test_instance.cutoff_idx-1:test_instance.cutoff_idx, :]
    #   should_predict = should_predict[:, :, test_instance.cutoff_idx-1:test_instance.cutoff_idx, :] # [b, a, t, 1]

    # else:
    #   prediction = prediction[:, :, :cutoff_idx, :]
    #   target = target[:, :, int(model_p.num_history_steps):int(model_p.num_history_steps+1), :]
    #   should_target = should_predict[:, :, int(model_p.num_history_steps):int(model_p.num_history_steps+1), :]
    #   should_predict = should_predict[:, :, :cutoff_idx, :]

    # should_target = tf.broadcast_to(should_target, target.shape).numpy()
    # should_plot_pred = tf.broadcast_to(should_predict[..., tf.newaxis, :], prediction.shape).numpy()

    # target_plot = tf.reshape(tf.boolean_mask(target, should_target), (-1,2)).numpy()
    # pred_plot = tf.reshape(tf.boolean_mask(prediction, should_plot_pred), (-1,2)).numpy()

    # ax0.scatter(pred_plot[::n,0], pred_plot[::n,1], label='prediction', color=blue, marker='.', s=1)
    # ax0.scatter(target_plot[::6*n,0], target_plot[::6*n,1], label='Target', color=red, marker='x', s=4)
    # ax0.set_title(f'{step}')
    # camera.snap()
    step += 1
    # if step >= 10:
    #   break

    # [b, a, t, n, 3] -> [b, a, t, n, 1]
    # norm(dx,dy)



    # input("next batch")
    # print(prediction.numpy()[0])
    # pred_list.append(prediction.numpy())
    # groundtrue_list.append(output_batch['agents/position'].numpy())
    # mixture_logits_list.append(scene_mixture_logits.numpy())

    # tqdm.tqdm.write(f'{scene_t},{prediction.shape}')

  # print(pred_list[:2])

  scene_id = scene_id.numpy()[0].decode("utf-8")
  print(scene_id)

  vis_path = '/home/user/socialnav/human-scene-transformer/tmp_vis'
  vis_data_path = os.path.join(vis_path, scene_id)


  # print(f"Saving {scene_id} visual data to {vis_data_path}")
  # groundtrue_list, groundtrue_list = tf.saved_model.load(vis_data_path)
  plt.title(scene_id)
  plot_iter = 0
  for gt_c, p_f, rob, rob_ori in tqdm.tqdm(zip(groundtrue_list, pred_list, robot_list, robot_orientation_list)):
    gt_c_xy = tf.reshape(gt_c, (-1,2))   # [b, a-, 1, 2]
    p_f_xy = tf.reshape(p_f, (-1,2))    # [b, a-, t-, n, 2]
    rob_xy = tf.reshape(rob, (-1,2))    # [b, 1, 2]
    rob_degree = tf.reshape(rob_ori, (1)) * 180 / np.pi

    ax0.scatter(p_f_xy[...,0], p_f_xy[...,1], label='Prediction' if plot_iter == 0 else "", color=blue, marker='.', s=5)
    ax0.scatter(gt_c_xy[...,0], gt_c_xy[...,1], label='Target' if plot_iter == 0 else "", color=red, marker='*', s=10)

    ax0.scatter(rob_xy[...,0], rob_xy[...,1], label='Robot' if plot_iter == 0 else "", color=robot_color, marker=(3,0,rob_degree), s=40)

    ax0.legend()

    camera.snap()
    plot_iter += 1
    # ax0.legend=[]


  # plt.show()
  anim = camera.animate(blit=True)
  print(f'saving {scene_id}_vis.mp4')
  anim.save(os.path.join(vis_path, f'{scene_id}_vis.mp4'))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(_MODEL_PATH.value, 'params', 'operative_config.gin')],
      None,
      skip_unknown=True)
  print('Actual gin config used:')
  print(gin.config_str())

  evaluation(_CKPT_PATH.value)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  app.run(main)
