dataset_info = dict(
    dataset_name='jump_broadcast',
    paper_info=dict(
        author='Katja Ludwig and Julian Lorenz and Robin Schön and Rainer Lienhart',
        title='All Keypoints You Need: Detecting Arbitrary Keypoints on the Body of Triple, High, and Long Jump '
              'Athletes',
        year='2023'
    ),
    keypoint_info={
        0:
        dict(name='head',
             id=0,
             color=[51, 153, 255],
             type='upper',
             swap=''),
        1:
        dict(
            name='neck',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='rsho',
            id=2,
            color=[200, 0, 0],
            type='upper',
            swap='lsho'),
        3:
        dict(
            name='relb',
            id=3,
            color=[200, 0, 0],
            type='upper',
            swap='lelb'),
        4:
        dict(
            name='rwri',
            id=4,
            color=[200, 0, 0],
            type='upper',
            swap='lwri'),
        5:
        dict(
            name='rhan',
            id=5,
            color=[200, 0, 0],
            type='upper',
            swap='lhan'),
        6:
        dict(
            name='lsho',
            id=6,
            color=[0, 240, 0],
            type='upper',
            swap='rsho'),
        7:
        dict(
            name='lelb',
            id=7,
            color=[0, 240, 0],
            type='upper',
            swap='relb'),
        8:
        dict(
            name='lwri',
            id=8,
            color=[0, 240, 0],
            type='upper',
            swap='rwri'),
        9:
        dict(
            name='lhan',
            id=9,
            color=[0, 240, 0],
            type='upper',
            swap='rhan'),
        10:
        dict(
            name='rhip',
            id=10,
            color=[200, 0, 0],
            type='lower',
            swap='lhip'),
        11:
        dict(
            name='rkne',
            id=11,
            color=[200, 0, 0],
            type='lower',
            swap='lkne'),
        12:
        dict(
            name='rank',
            id=12,
            color=[200, 0, 0],
            type='lower',
            swap='lank'),
        13:
        dict(
            name='rhee',
            id=13,
            color=[200, 0, 0],
            type='lower',
            swap='lhee'),
        14:
        dict(
            name='rtoe',
            id=14,
            color=[200, 0, 0],
            type='lower',
            swap='ltoe'),
        15:
        dict(
            name='lhip',
            id=15,
            color=[0, 240, 0],
            type='lower',
            swap='rhip'),
        16:
        dict(
            name='lkne',
            id=16,
            color=[0, 240, 0],
            type='lower',
            swap='rkne'),
        17:
        dict(
            name='lank',
            id=17,
            color=[0, 240, 0],
            type='lower',
            swap='rank'),
        18:
        dict(
            name='lhee',
            id=18,
            color=[0, 240, 00],
            type='lower',
            swap='rhee'),
        19:
        dict(
            name='ltoe',
            id=19,
            color=[0, 240, 0],
            type='lower',
            swap='rtoe')
    },
    skeleton_info={
        0:
        dict(link=('head', 'neck'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('neck', 'rsho'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('rsho', 'relb'), id=2, color=[200, 0, 0]),
        3:
        dict(link=('relb', 'rwri'), id=3, color=[200, 0, 0]),
        4:
        dict(link=('rwri', 'rhan'), id=4, color=[200, 0, 0]),
        5:
        dict(link=('neck', 'lsho'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('lsho', 'lelb'), id=6, color=[0, 240, 0]),
        7:
        dict(link=('lelb', 'lwri'), id=7, color=[0, 240, 0]),
        8:
        dict(link=('lwri', 'lhan'), id=8, color=[0, 240, 0]),
        9:
        dict(link=('rsho', 'rhip'), id=9, color=[200, 0, 0]),
        10:
        dict(link=('rhip', 'rkne'), id=10, color=[200, 0, 0]),
        11:
        dict(link=('rkne', 'rank'), id=11, color=[200, 0, 0]),
        12:
        dict(link=('rank', 'rhee'), id=12, color=[200, 0, 0]),
        13:
        dict(link=('rhee', 'rtoe'), id=13, color=[200, 0, 0]),
        14:
        dict(link=('lsho', 'lhip'), id=14, color=[0, 240, 0]),
        15:
        dict(link=('lhip', 'lkne'), id=15, color=[0, 240, 0]),
        16:
        dict(link=('lkne', 'lank'), id=16, color=[0, 240, 0]),
        17:
        dict(link=('lank', 'lhee'), id=17, color=[0, 240, 0]),
        18:
        dict(link=('lhee', 'ltoe'), id=18, color=[0, 240, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10
    ])

