//
//  ImagePicker.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 27.02.21.
//

import SwiftUI
import PhotosUI
import PromiseKit

struct ImagePicker: UIViewControllerRepresentable {
    
    
    @Binding var images: [UIImage]
    @Binding var showPicker: Bool
    
    // MARK: configuration
    
    // config of the filter, which type can be picked inside the picker
    var filter: PHPickerFilter = .images
    // config of the limited, how many items can be picked at once
    var limit: Int = 3
    
    // closure which returns true if items/images where selected
    var didFinishPicking: (_ didSelectItems: Bool) -> Void
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        
        var configuration = PHPickerConfiguration()
        configuration.filter = filter
        configuration.selectionLimit = limit
        
        // adding the configuration to the picker
        let picker = PHPickerViewController(configuration: configuration)
        
        // adding the delegate
        picker.delegate = context.coordinator
        
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {
        // needed for protocol but not in this case
    }
    
    func makeCoordinator() -> Coordinator {
        return ImagePicker.Coordinator(parent: self)
    }
    
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        
        var parent: ImagePicker
        
        init(parent: ImagePicker) {
            self.parent = parent
        }
        
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            
            // closing the view of the picker
            parent.showPicker.toggle()
            
            // getting the results
            firstly {
                when(fulfilled: results.compactMap(loop)).done {
                    // if we have results push the to the closure
                    self.parent.didFinishPicking(results.count > 0)
                    
                }
                
            }.catch { error in
                print(error.localizedDescription)
            }
        }
        
        func loop(img: PHPickerResult) -> Promise<Void> {
            return Promise { seal in
                print("Img: \(img)")
                // error handling, if items can be loaded
                if img.itemProvider.canLoadObject(ofClass: UIImage.self) {
                    
                    img.itemProvider.loadObject(ofClass: UIImage.self) { (image, error) in
                        if let error = error {
                            seal.reject(error)
                        }else {
                            if let img = image {
                                self.parent.images.append(img as! UIImage)
                                seal.fulfill(())
                            }
                        }
                    }
                }
            }
        }
    }
}
