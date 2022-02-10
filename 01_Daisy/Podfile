# Uncomment the next line to define a global platform for your project
# platform :ios, '9.0'

target 'Meet Me' do
  # Comment the next line if you don't want to use dynamic frameworks
  use_frameworks!
  
  # Pods for Meet Me
pod 'Firebase/Analytics', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
pod 'Firebase/Auth', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
pod 'Firebase/Storage', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
pod 'Firebase/Firestore', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
	# add the Firebase pod for Google Analytics
pod 'FirebaseFirestoreSwift', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
pod 'Firebase/Messaging', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
pod 'GeoFire/Utils', :git => 'https://gitlab.com/phlpp77/meet-me.git', :branch => 'DB-dev'
# add pods for any other desired Firebase products
# https://firebase.google.com/docs/ios/setup#available-pods


end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '9.2'
    end
  end
end




