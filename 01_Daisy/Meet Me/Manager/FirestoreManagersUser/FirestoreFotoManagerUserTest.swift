//
//  FirestoreFotoManagerUserTest.swift
//  Meet Me
//
//  Created by Lukas Dech on 22.02.21.
//

import Foundation
import Firebase
import FirebaseFirestoreSwift
import URLImage
import PromiseKit


class FirestoreFotoManagerUserTest: ObservableObject {
    
    let storage = Storage.storage()
    private var db: Firestore
    private var storageIds: [String] = []

    
    
    init() {
        db = Firestore.firestore()
    }
    
    
    // MARK: - Functions to save User Photos to Storage and the URL to Firestore
    

    
    
    func resizeImage(originalImage: UIImage?) -> Promise<Data> {
        return Promise { seal in
            if let originalImage = originalImage {
                    if let resizedImage = originalImage.resized(width: 350) {
                        if let data = resizedImage.pngData() {
                            seal.fulfill(data)
                        }
                    }
                }
            }
            
        }
    
    
    //Wird nicht direkt aufgerufen -> wird in savePhoto aufgerufen
    func uploadUserPhoto(data: Data) -> Promise<URL> {
        return Promise { seal in

            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            let imageName = UUID().uuidString
            storageIds.append(imageName)
            let storageRef = storage.reference()
            let photoRef = storageRef.child("UserImages2/\(currentUser.uid)/\(imageName).png")

            photoRef.putData(data, metadata: nil) { metadata, error in
                
                if let err = error {
                    seal.reject(err)
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
    
    func saveStorageIds(fotoPlace: Int) ->Promise<Void> {
        return Promise{ seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            var counter = fotoPlace
            let dbRef = db.collection("users").document(currentUser.uid)
            let _ = storageIds.compactMap { id in
                dbRef.updateData(["userPhotosId.\(counter)" : id ])
                counter = counter + 1
                return
                
            }
            storageIds = []
            seal.fulfill(())
        }
    }
    

        
    
    
    
    //Wird nicht direkt aufgerufen -> wird in savePhoto aufgerufen
    func savePhotoUrlToFirestore(url: URL, fotoPlace: Int) ->Promise<Void>{
        return Promise { seal in
            
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
                let _ =  db.collection("users")
                    .document(currentUser.uid).updateData(["userPhotos.\(fotoPlace)" : url.absoluteString]) { error in
                        if let error = error {
                            seal.reject(error)
                        }else {
                            seal.fulfill(())
                        }
                    }
            }

        }
    
    func uploadStockPhotoAsProfilPhoto(userModel : UserModel) ->Promise<URL> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            var stockUrl = stockURL
            switch userModel.gender {
                case "Male":
                    stockUrl = stockMale
                case "Female":
                    stockUrl = stockFemale
                default:
                    stockUrl = stockURL
            }
            let _ = db.collection("users").document(currentUser.uid).updateData(["userPhotos.0": stockUrl.absoluteString,
                                                                                 "userPhotosId.0": "StockPhoto"]) { error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(stockUrl)
                }
            }
        }
    }
    
    
    func reOrderPictures(position:Int, position1:Int, url1: String, urlId1 : String) ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            let _ = self.db.collection("users").document(currentUser.uid)
                .updateData(["userPhotos.\(position)" : url1,
                             "userPhotosId.\(position)" : urlId1,
                             "userPhotos.\(position1)" : FieldValue.delete(),
                             "userPhotosId.\(position1)" : FieldValue.delete()]) { error in
                    if let error = error {
                        seal.reject(error)
                    }else {
                        seal.fulfill(())
                    }
                }
        }
    }

    
    
    func changedProfilPicture(newProfilePicture: URL) ->Promise<Void> {
        return Promise { seal in
            print("changed ProfilPicture aufgeerufen")
            print("url : \(newProfilePicture)")
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
             db.collection("events").whereField("userId", isEqualTo: currentUser.uid).getDocuments{ (snapshot, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    let ids: [String]! = snapshot?.documents.compactMap { doc in
                        return doc.documentID
                    }
                    print("ids \(ids!)")
                    for id in ids {
                        self.db.collection("events").document(id).updateData(["profilePicture" : newProfilePicture.absoluteString]) { error in
                            if let error = error {
                                seal.reject(error)
                            }
                        }
                    }
                    seal.fulfill(())
                    
                }
            }

        }
        
    }

 
    func deleteImageFromStorage(storageId: String?) ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            print("delete Storage aufgerufen")
            if storageId != nil {
            let storageRef = storage.reference()
                let photoRef = storageRef.child("UserImages2/\(currentUser.uid)/\(storageId!).png")
            
            photoRef.delete()
                } else {
                    seal.fulfill(())
                }
            seal.fulfill(())
            }
            
            
        }
    
    
    
    func deleteImageFromUser(fotoPlace: Int) -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            let _ = db.collection("users").document(currentUser.uid)
                .updateData(["userPhotos.\(fotoPlace)" : FieldValue.delete(),
                             "userPhotosId.\(fotoPlace)" : FieldValue.delete()]) { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
            
        }
    }
    
    
    
}
    


        
    
    
    
    
