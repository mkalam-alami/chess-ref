import { describe, expect, it } from 'vitest';
import { resolveCameraId, videoInputs } from '../src/camera';

const dev = (kind: string, deviceId: string, label = '') => ({ kind: kind as MediaDeviceKind, deviceId, label });

describe('videoInputs', () => {
  it('keeps video inputs only and falls back to "Camera N" labels', () => {
    const cams = videoInputs([
      dev('audioinput', 'a1', 'Mic'),
      dev('videoinput', 'v1', 'camera2 0, facing back'),
      dev('videoinput', 'v2', '  '),
      dev('audiooutput', 'o1', 'Speaker'),
      dev('videoinput', 'v3'),
    ]);
    expect(cams).toEqual([
      { deviceId: 'v1', label: 'camera2 0, facing back' },
      { deviceId: 'v2', label: 'Camera 2' },
      { deviceId: 'v3', label: 'Camera 3' },
    ]);
  });

  it('drops entries without a deviceId (before permission)', () => {
    expect(videoInputs([dev('videoinput', ''), dev('videoinput', '')])).toEqual([]);
  });
});

describe('resolveCameraId', () => {
  const cams = [
    { deviceId: 'a', label: 'A' },
    { deviceId: 'b', label: 'B' },
  ];
  it('returns null without a stored id', () => {
    expect(resolveCameraId(null, cams)).toBeNull();
    expect(resolveCameraId('', cams)).toBeNull();
  });
  it('keeps a stored id that is still listed', () => {
    expect(resolveCameraId('b', cams)).toBe('b');
  });
  it('drops a stale stored id', () => {
    expect(resolveCameraId('gone', cams)).toBeNull();
  });
  it('tries the stored id while the list is unknown', () => {
    expect(resolveCameraId('b', [])).toBe('b');
  });
});
