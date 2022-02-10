//
//  FirestoreFotoMangerEvent.swift
//  Meet Me
//
//  Created by Lukas Dech on 19.02.21.
//

import Foundation
import Firebase
import FirebaseFirestoreSwift
import URLImage
import PromiseKit


class FirestoreFotoManagerEventTest: ObservableObject {
    
    let storage = Storage.storage()
    private var db: Firestore
    var imageName = ""
    
//    @Published var photoModel: [PhotoModelObject] = []
//    var stockPhotoModel: PhotoModel = PhotoModel()
//    var url: URL?

     
    init() {
          db = Firestore.firestore()
    }
    

    // MARK: - Functions to save User Photos to Storage and the URL to Firestore
    
    func resizeImage(originalImage: UIImage?) -> Promise<Data> {
        return Promise { seal in
            if let originalImage = originalImage {
                if let resizedImage = originalImage.resized(width: 260) {
                    if let data = resizedImage.pngData() {
                        seal.fulfill(data)
                    }
                }
            }
        }
    }
    
    //Wird nicht direkt aufgerufen -> wird in savePhoto aufgerufen
    func uploadEventPhoto(data: Data) -> Promise<URL> {
        return Promise { seal in
            
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            imageName = UUID().uuidString
            let storageRef = storage.reference()
            let photoRef = storageRef.child("EventImages2/\(currentUser.uid)/\(imageName).png")
            
            photoRef.putData(data, metadata: nil) { metadata, error in
                
                if let error = error {
                    seal.reject(error)
                    //print(err.localizedDescription)
                }
                photoRef.downloadURL { (url, error) in
                    
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(url!)
                    }
                }
            }
        }
        
    }
    
    
    //Wird nicht direkt aufgerufen -> wird in savePhoto aufgerufen
    func saveEventPhotoUrlToFirestore(url: URL,eventModel: EventModel) -> Promise<Void> {
        return Promise { seal in
            var checkUrl = url
            if checkUrl == stockURL || checkUrl.absoluteString ==  ""  {
                    checkUrl = stockURL
                }
            let _ = db.collection("events")
                .document(eventModel.eventId)
                .updateData(["pictureURL" : checkUrl.absoluteString,
                             "eventPhotosId" : imageName]) { error in
                    if let error = error {
                        seal.reject(error)
                    }else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    func deleteImageFromStorage(storageId: String?) ->Promise<Void> {
        return Promise { seal in
            print("delete Storage aufgerufen")
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            if storageId != nil {
            let storageRef = storage.reference()
                let photoRef = storageRef.child("EventImages2/\(currentUser.uid)/\(storageId!).png")
            photoRef.delete()
                } else {
                    seal.fulfill(())
                }
            seal.fulfill(())
            }
        }
}
    
    
