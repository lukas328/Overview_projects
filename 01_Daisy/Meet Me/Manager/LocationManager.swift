//
//  LocationManager.swift
//  Meet Me
//
//  Created by Lukas Dech on 07.03.21.
//

import Foundation
import CoreLocation
import MapKit
import PromiseKit

class LocationManager: NSObject, ObservableObject {
    
    private let locationManager = CLLocationManager()
    @Published var location: CLLocation? = nil
    @Published var region: MKCoordinateRegion = MKCoordinateRegion.defaultRegion
    
        override init() {
        super.init()
        
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = kCLDistanceFilterNone
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        
    }
}

extension LocationManager: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {

            guard let location = locations.last else {return}
            
            DispatchQueue.main.async {
                self.location = location
                self.region = MKCoordinateRegion(center: location.coordinate, latitudinalMeters: 100, longitudinalMeters: 100)

            }
        }
    }


extension MKCoordinateRegion {

    static var defaultRegion: MKCoordinateRegion {
        MKCoordinateRegion(center: CLLocationCoordinate2D.init(latitude: 10, longitude: 10),latitudinalMeters: 100, longitudinalMeters: 100)

    }


}
//(latitude: 53.5466399, longitude: 9.93079)



