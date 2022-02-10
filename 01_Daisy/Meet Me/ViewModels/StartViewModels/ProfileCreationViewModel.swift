//
//  ProfileCreationModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 29.01.21.
//

import Foundation
import Firebase
import FirebaseFirestoreSwift
import PromiseKit
import MapKit
import GeoFire

class ProfileCreationModel: ObservableObject {
    
    private var firestoreManagerUserTest: FirestoreManagerUserTest
    private var firestoreFotoMangerUserTest: FirestoreFotoManagerUserTest
    private var locationManager: LocationManager = LocationManager()
    @Published var saved: Bool = false
    @Published var message: String = ""
    private var counter = 0
    
    var userId: String = ""
    var name: String = "Name"
    var birthdayDate: Date = Date()
    var gender: String = "Gender"
    var searchingFor: String = ""
    var startProcessDone: Bool = true
    var userToken: String = ""
    
    let storage = Storage.storage()
    
    
    
    
    init() {
        firestoreManagerUserTest = FirestoreManagerUserTest()
        firestoreFotoMangerUserTest = FirestoreFotoManagerUserTest()
    }
    
    func askLocation() {
        _ = locationManager.$location.sink { location in
            _ = MKCoordinateRegion(center: location?.coordinate ?? CLLocationCoordinate2D(), latitudinalMeters: 200, longitudinalMeters: 200)
        }
    }


    
    func createUser(images: [UIImage]?) {
        guard let currentUser = Auth.auth().currentUser else {
            return
        }
        

//        self.birthdayDate = convertStringToDate(date: bDate)
        
        let userModel = UserModel(userId: currentUser.uid, name: name, birthdayDate: birthdayDate, gender: gender, startProcessDone: startProcessDone, searchingFor : searchingFor, userPhotos: [0: stockURL.absoluteString], userPhotosId: [0: "stockPhoto"], radiusInKilometer: 150, refreshCounter: 0, userStatus: "normal",reports: 0, loginToken: userToken, lastLogin: getShortDate())
        
        firstly {
            self.firestoreManagerUserTest.saveUser(userModel: userModel)
        }.then {
            self.firestoreManagerUserTest.createLikedEventsArray()
        }.then {
            self.firestoreManagerUserTest.createReportedEventsArray()
        }.done {
            if images != nil {
                when(fulfilled: images!.compactMap(self.uploadUserPhotos)).catch { error in
                    print(error)
                }
            }
        }.catch { error in
            print("DEBUG: catch, fehler in event creation \(error)")
            print(error.localizedDescription)
        }
    }

    
    func uploadUserPhotos(originalImage: UIImage) -> Promise<Void> {
        return Promise { seal in
            firstly{
                self.firestoreFotoMangerUserTest.resizeImage(originalImage: originalImage)
            }.then { picture in
                self.firestoreFotoMangerUserTest.uploadUserPhoto(data: picture)
            }.map { url in
                self.firestoreFotoMangerUserTest.savePhotoUrlToFirestore(url: url, fotoPlace: self.counter).cauterize()
                self.counter = self.counter + 1
            }.done {
                //self.counter = self.counter + 1
                _ = self.firestoreFotoMangerUserTest.saveStorageIds(fotoPlace: 0)
                seal.fulfill(())
            }.catch { error in
                seal.reject(error)
                
            }
        }
    }
    
    
    
    func convertStringToDate(date: String) -> Date {
        let dateFormatter = DateFormatter()
        
        dateFormatter.locale = Locale(identifier: "en_DE")
        dateFormatter.dateFormat = "dd/MM/yyyy"
        print(date)
        let date = dateFormatter.date(from: date)!
        return date
    }

}
