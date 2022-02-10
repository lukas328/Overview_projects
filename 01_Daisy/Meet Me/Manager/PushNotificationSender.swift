//
//  PushNotificationSender.swift
//  Meet Me
//
//  Created by Lukas Dech on 13.03.21.
//
import Firebase
import PromiseKit
import UIKit
class PushNotificationSender {
    

    func sendPushNotification(to token: String, title: String, body: String) {

        let urlString = "https://fcm.googleapis.com/fcm/send"
        let url = NSURL(string: urlString)!
        let paramString: [String : Any] = ["to" : token,
                                           "notification" : ["title" : title, "body" : body],
                                           "data" : ["user" : "test_id"]
        ]
        
        let request = NSMutableURLRequest(url: url as URL)
        request.httpMethod = "POST"
        request.httpBody = try? JSONSerialization.data(withJSONObject:paramString, options: [.prettyPrinted])
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("key=AAAAdTGBNWM:APA91bH_sCDcIS8FNjzYqmuUBtbu70Pw6Ymt1LHUsmXd3cIp-4vLxLQy54ZIn2VlVZeHVr0gtSQ7SGlO7U-oEKadRhp1lsF7zayV2XD4Hx8-wUnKtD7bc92gRosPA82aC2VGP_zHwPk_", forHTTPHeaderField: "Authorization")
        let task =  URLSession.shared.dataTask(with: request as URLRequest)  { (data, response, error) in
            do {
                if let jsonData = data {
                    if let jsonDataDict  = try JSONSerialization.jsonObject(with: jsonData, options: JSONSerialization.ReadingOptions.allowFragments) as? [String: AnyObject] {
                        NSLog("Received data:\n\(jsonDataDict))")
                    }
                }
            } catch let err as NSError {
                print("DEBUG: error in Push Notification")
                print(err.debugDescription)
            }
        }
        task.resume()
    }
}
    
    
    
