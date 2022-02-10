//
//  ImageHandler.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 17.04.21.
//

import SwiftUI

struct ImageHandler: View {
    
    @State var showPHPicker: Bool = true
    @State var showImageCropper: Bool = false
    
    @State var imagesFromPHPicker: [UIImage] = [UIImage(named: "cafe")!]
    @State private var tempInputImage: UIImage?
    
    @Binding var croppedImage: UIImage?
    @Binding var showView: Bool
    
    var body: some View {
        if showPHPicker {
            
            // image Picker
            ImagePicker(images: $imagesFromPHPicker, showPicker: $showPHPicker, limit: 1) { (imagesPicked) in
                if imagesPicked {
                    // show image cropper sheet
                    tempInputImage = imagesFromPHPicker.last
                    self.showImageCropper = true
                    
//                    // clear the array for further picking and re-picking of images
//                    imagesFromPHPicker = []
                    
                    
                } else {
                    
                    showView = false
                    
                }
            }
                        
            
            
        }
        
        if showImageCropper {
            ImageCropper(image: $tempInputImage, visible: $showView) { (croppedImage) in

                self.croppedImage = croppedImage
                showView = false
                
            }
            .background(
                Color.primary
                    .ignoresSafeArea()
            )
        }
    }
    
}


//struct ImageHandler_Previews: PreviewProvider {
//    static var previews: some View {
//        ImageHandler()
//    }
//}
