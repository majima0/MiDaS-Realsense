# requirements
## Realsense SDK
### https://github.com/IntelRealSense/librealsense
## OpenCV455
### https://opencv.org/releases/
## ONNX
### https://github.com/isl-org/MiDaS/releases/tag/v2_1
### Download model-.onnx

# depth estimetion
これは深度推定とリアルセンスの深度カメラを同時に表示するリポジトリです。深度推定はMiDaSのモデルを使用しています。コードを書き換えることでPCのCPUを使用したものや、使うモデルを変更することができます。コード内では単眼深度推定のコードもコメントアウトしているので、使ってみてください。