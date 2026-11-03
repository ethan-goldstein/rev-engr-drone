# Drone voiceover (~30 s): the defense system pitch

Calm, smooth, almost bored female AI voice, like a corporate city-AI ad. A beat of silence after each
line; let "It patrols." and "Defense. Automated." hang.

```
Your home has walls. Locks. Cameras that only watch.
This doesn't just watch.
It patrols.

It knows who belongs... and who doesn't.
An unknown face at the perimeter. Detected. Locked. Followed.

Wherever they run, it follows.
Every step, recorded. Every second, logged.
You're alerted before they reach the door.

No guard. No pilot. No blind spots.
Defense. Automated.

Sentinel OS. Online.
```

"Sentinel OS" is a placeholder name for your system. Swap in your own.

## Where each line goes in the video
| Line | Show on screen |
|---|---|
| Walls / locks / cameras that only watch | slow shots of a house, a door lock, a doorbell camera |
| This doesn't just watch / it patrols | the drone lifting off, wide tripod shot |
| Knows who belongs / unknown face | the drone's camera view, boxes appearing on people |
| Detected / locked / followed | the FACE LOCK card snapping on, the drone turning to follow |
| Wherever they run / every step recorded | someone walking away with the drone trailing; the telemetry graphs |
| Alerted before they reach the door | the laptop screen with the lock and countdown chips |
| No guard / no pilot / no blind spots | the rainbow 3D map spinning (`map.mp4`) |
| Defense. Automated. / Sentinel OS | black screen, title text |

## Draft audio
- `voiceover_fx.m4a`: the macOS Samantha voice with a digital, slightly metallic effect.
- `voiceover_clean.m4a`: the same voice with no effect, for adding your own (CapCut: Voice effects → Robot).
- `voiceover_say.txt`: the script with pause marks. To re-record after editing it:
  ```bash
  (echo "[[rate 185]]"; cat voiceover_say.txt) | say -v Samantha -o voice.aiff
  ```
