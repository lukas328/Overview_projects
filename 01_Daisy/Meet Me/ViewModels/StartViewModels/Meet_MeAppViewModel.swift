//
//  Meet_MeAppViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 14.02.21.
//
//
import Foundation
import Firebase
import FirebaseFirestoreSwift
import FirebaseAuth

class Meet_MeAppViewModel: ObservableObject {
    var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()


    func CheckUserAccForAutoLogin() -> Bool{
        if Auth.auth().currentUser != nil{
            return true
            
        } else {
            return false

        }
    }

}
