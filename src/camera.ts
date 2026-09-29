/** A displayable, drawable frame source (live camera or a looping file). */
export interface FrameSource {
  /** Element to mount in the page; sized by CSS with object-fit: cover. */
  readonly element: HTMLVideoElement | HTMLCanvasElement;
  /** Native size of the source in pixels (0 until known). */
  readonly width: number;
  readonly height: number;
  isReady(): boolean;
  stop(): void;
}

export type CameraErrorKind = 'insecure' | 'unsupported' | 'denied' | 'no-camera' | 'other';

export class CameraError extends Error {
  constructor(
    readonly kind: CameraErrorKind,
    message: string,
  ) {
    super(message);
  }
}

const MESSAGES: Record<CameraErrorKind, string> = {
  insecure: 'The camera needs a secure connection. Open this page over HTTPS.',
  unsupported: 'This browser does not support camera access.',
  denied: 'Camera permission was denied. Allow camera access in the browser settings and try again.',
  'no-camera': 'No camera was found on this device.',
  other: 'The camera could not be started.',
};

export function cameraError(kind: CameraErrorKind, detail?: string): CameraError {
  return new CameraError(kind, detail ? `${MESSAGES[kind]} (${detail})` : MESSAGES[kind]);
}

function makeVideo(): HTMLVideoElement {
  const v = document.createElement('video');
  v.className = 'media';
  v.playsInline = true;
  v.muted = true;
  v.autoplay = true;
  v.setAttribute('playsinline', '');
  v.setAttribute('muted', '');
  return v;
}

class VideoSource implements FrameSource {
  constructor(
    readonly element: HTMLVideoElement,
    private onStop: () => void,
  ) {}
  get width(): number {
    return this.element.videoWidth;
  }
  get height(): number {
    return this.element.videoHeight;
  }
  isReady(): boolean {
    return this.element.readyState >= 2 && this.width > 0;
  }
  stop(): void {
    this.onStop();
    this.element.remove();
  }
}

class CameraVideoSource extends VideoSource implements CameraSource {
  constructor(
    element: HTMLVideoElement,
    onStop: () => void,
    readonly deviceId: string | null,
    readonly fellBack: boolean,
  ) {
    super(element, onStop);
  }
}

class ImageSource implements FrameSource {
  constructor(
    readonly element: HTMLCanvasElement,
  ) {}
  get width(): number {
    return this.element.width;
  }
  get height(): number {
    return this.element.height;
  }
  isReady(): boolean {
    return true;
  }
  stop(): void {
    this.element.remove();
  }
}

/** A video input as shown in the camera picker. */
export interface CameraInfo {
  deviceId: string;
  label: string;
}

/**
 * Video inputs from an enumerateDevices() result, with a "Camera N" label when the browser hides labels
 * (before permission is granted). Entries without a deviceId (also pre-permission) are dropped.
 */
export function videoInputs(devices: ReadonlyArray<Pick<MediaDeviceInfo, 'kind' | 'deviceId' | 'label'>>): CameraInfo[] {
  const out: CameraInfo[] = [];
  for (const d of devices) {
    if (d.kind !== 'videoinput' || !d.deviceId) continue;
    out.push({ deviceId: d.deviceId, label: d.label.trim() || `Camera ${out.length + 1}` });
  }
  return out;
}

/**
 * The stored camera to request, or null for the default rear camera. With a known (non-empty) camera list a
 * stored id that is not in it is stale (ids can change) and resolves to null; with an unknown list it is tried.
 */
export function resolveCameraId(stored: string | null, cameras: readonly CameraInfo[]): string | null {
  if (!stored) return null;
  if (cameras.length === 0) return stored;
  return cameras.some((c) => c.deviceId === stored) ? stored : null;
}

/** Lists video inputs. Labels (and on some browsers ids) are only available once camera permission was granted. */
export async function listCameras(): Promise<CameraInfo[]> {
  try {
    return videoInputs(await navigator.mediaDevices.enumerateDevices());
  } catch {
    return [];
  }
}

const CAMERA_KEY = 'chess-ref.cameraId';

export function loadCameraId(): string | null {
  try {
    return localStorage.getItem(CAMERA_KEY);
  } catch {
    return null;
  }
}

export function saveCameraId(id: string | null): void {
  try {
    if (id) localStorage.setItem(CAMERA_KEY, id);
    else localStorage.removeItem(CAMERA_KEY);
  } catch {
    /* storage unavailable */
  }
}

/** A live camera source. */
export interface CameraSource extends FrameSource {
  /** deviceId of the active track, if the browser reports it. */
  readonly deviceId: string | null;
  /** True when a specific device was requested but could not be opened, so the default camera was used. */
  readonly fellBack: boolean;
}

const DEVICE_FALLBACK_ERRORS = ['OverconstrainedError', 'NotFoundError', 'NotReadableError', 'DevicesNotFoundError'];

function mapGetUserMediaError(e: unknown): CameraError {
  const name = e instanceof DOMException ? e.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return cameraError('denied');
  if (name === 'NotFoundError' || name === 'OverconstrainedError' || name === 'DevicesNotFoundError')
    return cameraError('no-camera');
  return cameraError('other', e instanceof Error ? e.message : String(e));
}

/** Starts the camera `deviceId`, or the default rear camera; falls back to the latter if the device fails. */
export async function startCamera(deviceId?: string | null): Promise<CameraSource> {
  if (!window.isSecureContext) throw cameraError('insecure');
  if (!navigator.mediaDevices?.getUserMedia) throw cameraError('unsupported');

  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const open = (video: MediaTrackConstraints) => navigator.mediaDevices.getUserMedia({ video, audio: false });
  let stream: MediaStream | null = null;
  let fellBack = false;
  if (deviceId) {
    try {
      stream = await open({ deviceId: { exact: deviceId }, ...size });
    } catch (e) {
      const name = e instanceof DOMException ? e.name : '';
      if (!DEVICE_FALLBACK_ERRORS.includes(name)) throw mapGetUserMediaError(e);
      console.warn(`camera ${deviceId} failed (${name}), using the default camera`);
      fellBack = true;
    }
  }
  if (!stream) {
    try {
      stream = await open({ facingMode: 'environment', ...size });
    } catch (e) {
      throw mapGetUserMediaError(e);
    }
  }
  const s = stream;

  await enableContinuousFocus(s);

  const video = makeVideo();
  video.srcObject = s;
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(cameraError('other', 'video error'));
  });
  await video.play().catch(() => undefined);
  const activeId = s.getVideoTracks()[0]?.getSettings().deviceId || null;
  return new CameraVideoSource(video, () => s.getTracks().forEach((t) => t.stop()), activeId, fellBack);
}

async function enableContinuousFocus(stream: MediaStream): Promise<void> {
  const track = stream.getVideoTracks()[0];
  if (!track) return;
  try {
    const caps = track.getCapabilities?.() as { focusMode?: string[] } | undefined;
    if (caps?.focusMode?.includes('continuous')) {
      await track.applyConstraints({ advanced: [{ focusMode: 'continuous' } as MediaTrackConstraintSet] });
    }
  } catch {
    // Focus control is best effort.
  }
}

/** Loads an image (shown as a still frame) or a video (looped) chosen by the user. */
export async function loadFileSource(file: File): Promise<FrameSource> {
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('video/')) {
    const video = makeVideo();
    video.loop = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(cameraError('other', 'cannot read video file'));
    });
    await video.play().catch(() => undefined);
    return new VideoSource(video, () => URL.revokeObjectURL(url));
  }
  if (file.type.startsWith('image/')) {
    try {
      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement('canvas');
      canvas.className = 'media';
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
      bitmap.close();
      return new ImageSource(canvas);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  URL.revokeObjectURL(url);
  throw cameraError('other', 'unsupported file type');
}

export interface GrabbedFrame {
  bitmap: ImageBitmap;
  width: number;
  height: number;
}

/** Grabs downscaled frames, allowing only one in flight; extra requests are dropped. */
export class FrameGrabber {
  private busy = false;

  constructor(public longSide = 640) {}

  get isBusy(): boolean {
    return this.busy;
  }

  /** Returns null if a frame is already in flight or the source is not ready. */
  async grab(source: FrameSource): Promise<GrabbedFrame | null> {
    if (this.busy || !source.isReady()) return null;
    this.busy = true;
    try {
      const scale = this.longSide / Math.max(source.width, source.height);
      const width = Math.max(1, Math.round(source.width * scale));
      const height = Math.max(1, Math.round(source.height * scale));
      const bitmap = await createImageBitmap(source.element, {
        resizeWidth: width,
        resizeHeight: height,
        resizeQuality: 'medium',
      });
      return { bitmap, width, height };
    } catch {
      this.busy = false;
      return null;
    }
  }

  /** Call when the worker has answered (or the frame was abandoned). */
  release(): void {
    this.busy = false;
  }
}
