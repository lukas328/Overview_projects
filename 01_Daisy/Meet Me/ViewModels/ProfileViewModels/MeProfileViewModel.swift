//
//  MeProfileModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 13.02.21.
//

import Foundation
import PromiseKit
import SwiftUI
import Firebase



class MeProfileViewModel: ObservableObject {
    private var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()
    private var firestoreFotoManagerUserTest: FirestoreFotoManagerUserTest = FirestoreFotoManagerUserTest()

    
    @Published var userModel: UserModel = stockUser
    @Published var genders: [String: Int] = ["Female": 0, "Male": 1, "All": 2]
    @Published var picked = 1
    
    
     var gender: [String] = ["Female", "Male", "All"]
    private var url: URL = stockURL
    private var db: Firestore
    private var listener: ListenerRegistration?
    
    
    
    init() {
        db = Firestore.firestore()
    }
    

    
    func getCurrentUser() {
        guard let currentUser = Auth.auth().currentUser else {
            return
        }


        listener = db.collection("users").document(currentUser.uid).addSnapshotListener { snapshot, error in
            if let error = error {
                print(error.localizedDescription)
            } else {
                if let snapshot = snapshot {
                    let userModel = try? snapshot.data(as: UserModel.self)
                    DispatchQueue.main.async {
                        if userModel != nil {
                            self.userModel = userModel!
                            self.picked = self.genders[userModel!.searchingFor]!
                        }else {
                            print(Err("cant get userProfil"))
                        }
                    }
                }
            }

        }
    }
    
    func changedSearchingFor(searchingFor: Int){
        if self.gender[searchingFor] != userModel.searchingFor {
            firstly {
                self.firestoreManagerUserTest.setSearchingForUserProfile(searchingFor: self.gender[searchingFor])
            }.then {
                self.firestoreManagerUserTest.setSearchingForEvents(searchingFor: self.gender[searchingFor])
            }.catch { error in
                print("DEBUG: Error in chagedSerachingFor: \(error)")
                print("DEBUG: Error localized in: \(error.localizedDescription)")
            }
        }
    }
    
    func changedRange(radius: Double){
        firstly {
            firestoreManagerUserTest.setRadius(radius: radius)
        }.catch { error in
            print("DEBUG: Error in changedRadius: \(error)")
            print("DEBUG: Error localized in: \(error.localizedDescription)")
            
        }
    }
    
    func addPhotoInPosition(image: UIImage, position: Int) {
        var checkPosition = position
        if userModel.userPhotosId[checkPosition - 1] == nil && position != 0{
            checkPosition = checkPosition - 1
        }
        
        firstly {
            self.firestoreFotoManagerUserTest.deleteImageFromStorage(storageId: self.userModel.userPhotosId[position])
        }.then {
            self.firestoreFotoManagerUserTest.resizeImage(originalImage: image)
        }.then { picture in
            self.firestoreFotoManagerUserTest.uploadUserPhoto(data: picture)
        }.map { url in
            self.firestoreFotoManagerUserTest.savePhotoUrlToFirestore(url: url, fotoPlace: checkPosition).cauterize()
            self.url = url
        }.done {
            _ = self.firestoreFotoManagerUserTest.saveStorageIds(fotoPlace: checkPosition)
        }.catch { error in
            print("DEBUG: error in addPhoto, error: \(error)")
            print("DEBUG: error localized \(error.localizedDescription)")
        }
        if position == 0 {
            firstly {
                firestoreFotoManagerUserTest.changedProfilPicture(newProfilePicture: self.url)
            }.catch { error in
                print("DEBUG: error in addPhoto, error: \(error)")
                print("DEBUG: error localized \(error.localizedDescription)")
            }
        }
    }
    
    func deletePhoto(position: Int) {
        //let position1 = position + 1
        let storageId =  userModel.userPhotosId[position]
        firstly {
            self.firestoreFotoManagerUserTest.deleteImageFromStorage(storageId: storageId)
        }.then {
            self.changePhoto(position: position)
        }.catch { error in
            print("DEBUG: error in deletePhoto error: \(error)")
            print("DEBUG: error localized: \(error.localizedDescription)")
        }
    }
    
    func changePhoto(position: Int) -> Promise<Void> {
        return Promise { seal in
            let position1 = position + 1
            let position2 = position + 2
            //Check if it is the last picture from User
            if self.userModel.userPhotosId[position1] != nil {
                //if not set the picture behind it on the position place and delete it on the old place
                let newUrl = userModel.userPhotos[position1]!
                firstly {
                    self.firestoreFotoManagerUserTest.reOrderPictures(position: position,
                                                                      position1: position1,
                                                                      url1: userModel.userPhotos[position1]!,
                                                                      urlId1: userModel.userPhotosId[position1]!)
                    //if the picture changed on place 0, change event pictures
                }.done {
                    if position == 0 {
                        firstly {
                            self.firestoreFotoManagerUserTest.changedProfilPicture(newProfilePicture: URL(string: newUrl)!)
                        }.catch { error in
                            seal.reject(error)
                        }
                    }
                    //if there are 3 pictures if true set set third picture on the second place
                    if self.userModel.userPhotosId[position2] != nil {
                        firstly {
                            self.firestoreFotoManagerUserTest.reOrderPictures(position: position1,
                                                                              position1: position2,
                                                                              url1: self.userModel.userPhotos[position2]!,
                                                                              urlId1: self.userModel.userPhotosId[position2]!)
                        }.catch { error in
                            seal.reject(error)
                        }
                    }
                }.catch { error in
                    seal.reject(error)
                }
            } else {
                //if it is the last picture from user
                firstly {
                    //delete the picture from user profil
                    self.firestoreFotoManagerUserTest.deleteImageFromUser(fotoPlace: position)
                    
                }.done {
                    if position == 0 {
                        firstly {
                            self.firestoreFotoManagerUserTest.uploadStockPhotoAsProfilPhoto(userModel: self.userModel)
                        }.then { stock in
                            self.firestoreFotoManagerUserTest.changedProfilPicture(newProfilePicture: stock)
                        }.catch { error in
                            seal.reject(error)
                        }
                    }
                }.catch { error in
                    seal.reject(error)
                }
            }
        }
    }
    
    
    
    func stopListening(){
        listener?.remove()
    }
}





