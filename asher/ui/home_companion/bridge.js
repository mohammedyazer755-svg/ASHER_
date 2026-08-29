// Presentation-only QWebChannel bridge for the Home Companion
(function() {
  'use strict';

  window.CompanionBridge = {
    state: 'STANDBY',
    audioLevel: 0.0,
    character: 'male',
    active: true,
    reducedMotion: false,
    qtBridge: null,

    init: function() {
      if (window.qt && window.qt.webChannelTransport && window.QWebChannel) {
        new window.QWebChannel(window.qt.webChannelTransport, function(channel) {
          window.CompanionBridge.qtBridge = channel.objects.bridge;
          var bridge = window.CompanionBridge.qtBridge;

          if (bridge) {
            bridge.stateChangedSignal.connect(function(state) {
              window.CompanionBridge.setState(state);
            });
            bridge.audioLevelSignal.connect(function(level) {
              window.CompanionBridge.setAudioLevel(level);
            });
            bridge.characterSignal.connect(function(charName) {
              window.CompanionBridge.setCharacter(charName);
            });
            bridge.activeSignal.connect(function(isActive) {
              window.CompanionBridge.setActive(isActive);
            });
            bridge.reducedMotionSignal.connect(function(reduced) {
              window.CompanionBridge.setReducedMotion(reduced);
            });

            // Initial queries
            if (typeof bridge.initialState === 'string') {
              window.CompanionBridge.setState(bridge.initialState);
            }
            if (typeof bridge.initialCharacter === 'string') {
              window.CompanionBridge.setCharacter(bridge.initialCharacter);
            }
            if (typeof bridge.initialAudioLevel === 'number') {
              window.CompanionBridge.setAudioLevel(bridge.initialAudioLevel);
            }
            if (typeof bridge.initialReducedMotion === 'boolean') {
              window.CompanionBridge.setReducedMotion(bridge.initialReducedMotion);
            }
            if (typeof bridge.initialActive === 'boolean') {
              window.CompanionBridge.setActive(bridge.initialActive);
            }
          }

          if (window.CompanionScene && window.CompanionScene.isReady) {
            window.CompanionBridge.notifyReady();
          }
        });
      }
    },

    notifyReady: function() {
      if (window.CompanionBridge.qtBridge && typeof window.CompanionBridge.qtBridge.rendererReady === 'function') {
        try {
          window.CompanionBridge.qtBridge.rendererReady();
        } catch (e) {
          console.warn('Could not call rendererReady on qtBridge', e);
        }
      }
    },

    notifyVrmMissing: function(gender) {
      if (window.CompanionBridge.qtBridge && typeof window.CompanionBridge.qtBridge.vrmMissing === 'function') {
        try {
          window.CompanionBridge.qtBridge.vrmMissing(String(gender));
        } catch (e) {
          console.warn('Could not call vrmMissing on qtBridge', e);
        }
      }
    },

    setState: function(state) {
      this.state = String(state || 'STANDBY').toUpperCase();
      if (window.CompanionScene) {
        window.CompanionScene.onStateChanged(this.state);
      }
    },

    setAudioLevel: function(level) {
      var clamped = Math.max(0.0, Math.min(1.0, Number(level) || 0.0));
      this.audioLevel = clamped;
      if (window.CompanionScene) {
        window.CompanionScene.onAudioLevel(clamped);
      }
    },

    setCharacter: function(characterName) {
      var name = String(characterName || 'male').toLowerCase();
      if (name !== 'male' && name !== 'female') {
        name = 'male';
      }
      this.character = name;
      if (window.CompanionScene) {
        window.CompanionScene.switchCharacter(name);
      }
    },

    setActive: function(isActive) {
      this.active = Boolean(isActive);
      if (window.CompanionScene) {
        window.CompanionScene.setActive(this.active);
      }
    },

    setReducedMotion: function(reduced) {
      this.reducedMotion = Boolean(reduced);
      if (window.CompanionScene) {
        window.CompanionScene.setReducedMotion(this.reducedMotion);
      }
    }
  };

  document.addEventListener('DOMContentLoaded', function() {
    window.CompanionBridge.init();
  });
})();
