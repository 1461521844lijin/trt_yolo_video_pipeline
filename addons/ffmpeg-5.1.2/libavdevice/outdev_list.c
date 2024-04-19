static const AVOutputFormat * const outdev_list[] = {
    &ff_alsa_muxer,
    &ff_fbdev_muxer,
    &ff_oss_muxer,
    &ff_sdl2_muxer,
    &ff_sndio_muxer,
    &ff_v4l2_muxer,
    &ff_xv_muxer,
    NULL };
