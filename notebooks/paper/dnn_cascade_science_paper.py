def _dumpScenario():
    from icecube.shovelart import ActivePixmapOverlay, Arrow, ArtistHandle, ArtistHandleList, ArtistKeylist, BaseLineObject, ChoiceSetting, ColorMap, ColoredObject, ConstantColorMap, ConstantFloat, ConstantQColor, ConstantTime, ConstantVec3d, Cylinder, DynamicLines, FileSetting, I3TimeColorMap, KeySetting, LinterpFunctionFloat, LinterpFunctionQColor, LinterpFunctionVec3d, OMKeySet, OverlayLine, OverlaySizeHint, OverlaySizeHints, ParticlePath, ParticlePoint, Phantom, PixmapOverlay, PyArtist, PyQColor, PyQFont, PyVariantFloat, PyVariantQColor, PyVariantTime, PyVariantVec3d, RangeSetting, Scenario, SceneGroup, SceneObject, SceneOverlay, SolidObject, Sphere, StaticLines, StepFunctionFloat, StepFunctionQColor, StepFunctionTime, StepFunctionVec3d, Text, TextOverlay, TimePoint, TimeWindow, TimeWindowColor, VariantFloat, VariantQColor, VariantTime, VariantVec3d, VariantVec3dList, Vec3dList, ZPlane, vec3d
    from icecube.icetray import OMKey
    from icecube.icetray import logging
    scenario = window.gl.scenario
    scenario.clear()
    try:
        artist = scenario.add( 'Bubbles', ['I3Geometry', 'InIceDSTPulses', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'scale', 10 )
        scenario.changeSetting( artist, 'colormap', I3TimeColorMap() )
        scenario.changeSetting( artist, 'power', 0.15 )
        scenario.changeSetting( artist, 'custom color window', '' )
        scenario.changeSetting( artist, 'log10(delay/ns)', 5 )
        scenario.changeSetting( artist, 'static', PyQColor(255,0,255,255) )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of Bubbles: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of Bubbles: " + str(e) )
    try:
        artist = scenario.add( 'Detector', ['I3Geometry', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'DOM color', PyQColor(115,115,115,255) )
        scenario.changeSetting( artist, 'DOM radius', 1 )
        scenario.changeSetting( artist, 'outline width', 0 )
        scenario.changeSetting( artist, 'high quality DOMs', True )
        scenario.changeSetting( artist, 'string cross', True )
        scenario.changeSetting( artist, 'string color', PyQColor(115,115,115,255) )
        scenario.changeSetting( artist, 'string width', 2 )
        scenario.changeSetting( artist, 'hide', 2 )
        scenario.changeSetting( artist, 'DOM labels', False )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of Detector: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of Detector: " + str(e) )
    try:
        artist = scenario.add( 'Ice', [] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, '3D dust', False )
        scenario.changeSetting( artist, 'Dust density', 1.5 )
        scenario.changeSetting( artist, 'Dust scatter', 0.2 )
        scenario.changeSetting( artist, 'Show bedrock', True )
        scenario.changeSetting( artist, 'Color ice', PyQColor(25,25,255,255) )
        scenario.changeSetting( artist, 'Color bedrock', PyQColor(128,102,102,255) )
        scenario.changeSetting( artist, 'Plane width', '2200 m' )
        scenario.changeSetting( artist, 'Show ice', True )
        scenario.changeSetting( artist, 'Line width', 1 )
        scenario.changeSetting( artist, 'Show dust', False )
        scenario.changeSetting( artist, 'Color dust', PyQColor(100,100,100,50) )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of Ice: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of Ice: " + str(e) )
    try:
        artist = scenario.add( 'TextSummary', ['EventGeneratorSelectedRecoNN_I3Particle', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'font', PyQFont.fromString('Ubuntu,11,-1,5,50,0,0,0,0,0') )
        scenario.changeSetting( artist, 'short', True )
        scenario.changeSetting( artist, 'fontsize', 11 )
        scenario.setOverlaySizeHints( artist, [OverlaySizeHint(20,136,478,193), ] )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of TextSummary: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of TextSummary: " + str(e) )
    try:
        artist = scenario.add( 'TextSummary', ['I3EventHeader', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'font', PyQFont.fromString('Ubuntu,11,-1,5,50,0,0,0,0,0') )
        scenario.changeSetting( artist, 'short', True )
        scenario.changeSetting( artist, 'fontsize', 11 )
        scenario.setOverlaySizeHints( artist, [OverlaySizeHint(14,16,344,85), ] )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of TextSummary: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of TextSummary: " + str(e) )
    try:
        artist = scenario.add( 'TextSummary', ['EventGeneratorSelectedRecoNNCircularUncertainty', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'font', PyQFont.fromString('Ubuntu,11,-1,5,50,0,0,0,0,0') )
        scenario.changeSetting( artist, 'short', True )
        scenario.changeSetting( artist, 'fontsize', 11 )
        scenario.setOverlaySizeHints( artist, [OverlaySizeHint(20,347,204,31), ] )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of TextSummary: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of TextSummary: " + str(e) )
    try:
        artist = scenario.add( 'Particles', ['EventGeneratorSelectedRecoNN_I3Particle', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'min. energy [track]', '' )
        scenario.changeSetting( artist, 'scale', 10 )
        scenario.changeSetting( artist, 'show light fronts', False )
        scenario.changeSetting( artist, 'colormap', I3TimeColorMap() )
        scenario.changeSetting( artist, 'power', 0.15 )
        scenario.changeSetting( artist, 'color', PyQColor(101,101,101,153) )
        scenario.changeSetting( artist, 'vertex size', 0 )
        scenario.changeSetting( artist, 'labels', True )
        scenario.changeSetting( artist, 'Cherenkov cone size', 0 )
        scenario.changeSetting( artist, 'blue light fronts', True )
        scenario.changeSetting( artist, 'incoming/outgoing', True )
        scenario.changeSetting( artist, 'color by type', True )
        scenario.changeSetting( artist, 'arrow head size', 100 )
        scenario.changeSetting( artist, 'linewidth', 3 )
        scenario.changeSetting( artist, 'min. energy [cascade]', '' )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of Particles: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of Particles: " + str(e) )
    try:
        artist = scenario.add( 'Position', ['EventGeneratorSelectedRecoNN_I3Particle_vertex', ] )
        scenario.setIsActive( artist, False )
        scenario.changeSetting( artist, 'color', PyQColor(255,255,255,255) )
        scenario.changeSetting( artist, 'size', 30 )
        scenario.setIsActive( artist, True )
    except StandardError as e:
        logging.log_error( e.__class__.__name__ + " occured while loading saved state of Position: " + str(e) )
    except:
        logging.log_error( "Unknown error occured while loading saved state of Position: " + str(e) )
    window.gl.setCameraPivot(7.17023658752, 36.1024436951, -63.2644119263)
    window.gl.setCameraLoc(642.906555176, 1550.77319336, 294.165283203)
    window.gl.setCameraOrientation(-0.92199587822, 0.386981546879, -7.0333480835e-06)
    window.gl.cameraLock = False
    window.gl.perspectiveView = True
    window.gl.backgroundColor = PyQColor(255,255,255,255)
    window.timeline.rangeFinder = "TimeWindow"
    window.frame_filter.code = ""
    window.activeView = 0
_dumpScenario()
del _dumpScenario