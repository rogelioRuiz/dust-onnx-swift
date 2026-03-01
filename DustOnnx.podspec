Pod::Spec.new do |s|
  s.name = 'DustOnnx'
  s.version = File.read(File.join(__dir__, 'VERSION')).strip
  s.summary = 'Standalone ONNX runtime session management and preprocessing for Dust.'
  s.license = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.homepage = 'https://github.com/rogelioRuiz/dust-onnx-swift'
  s.author = 'Techxagon'
  s.source = { :git => 'https://github.com/rogelioRuiz/dust-onnx-swift.git', :tag => s.version.to_s }

  s.source_files = 'Sources/DustOnnx/**/*.swift'
  s.module_name = 'DustOnnx'
  s.ios.deployment_target = '16.0'

  s.dependency 'DustCore'
  s.dependency 'onnxruntime-objc', '~> 1.20'
  s.swift_version = '5.9'
end
